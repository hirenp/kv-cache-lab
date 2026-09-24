"""KV Cache Lab V2: how decode latency grows with context length.

Reads top to bottom: preflight, measure memory bandwidth, then for each cache
type and context length fill the cache, time decode steps, and fit a line of
latency against KV bytes.
"""

import argparse
import csv
import statistics
import time

import torch
import transformers.integrations.sdpa_attention as sdpa_attention
from transformers import AutoModelForCausalLM, DynamicCache, StaticCache
from transformers.cache_utils import DynamicLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from hardware_info import print_hardware_info
from kv_lab import MODEL_ID, banner, kv_bytes_per_layer, mib, sync

WARMUP_STEPS = 5
APPLE_QUOTED_GBPS = 273
ORIGINAL_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]
KIND_SETS = {
    "both": ["dynamic", "static"],
    "all": ["dynamic", "dynamic-inplace", "static", "static-patched"],
}


def measure_bandwidth(device):
    # A large on-device copy reads and writes every byte once, so
    # 2 * size / time is the achieved memory traffic.
    x = torch.empty(2**29, dtype=torch.bfloat16, device=device)  # 1 GiB
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    sync(device)
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        y.copy_(x)
    sync(device)
    seconds = (time.perf_counter() - start) / iterations
    size = x.numel() * x.element_size()
    del x, y
    return 2 * size / seconds / 1e9


class InPlaceLayer(DynamicLayer):
    # Ablation for "dynamic-inplace": DynamicCache semantics, but preallocate the
    # buffer and write each new token into its slot instead of torch.cat-ing a
    # full copy of the cache on every step. keys/values are views of the filled part.
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        batch, heads, _, dim = key_states.shape
        self.key_buffer = key_states.new_empty(batch, heads, self.capacity, dim)
        self.value_buffer = value_states.new_empty(batch, heads, self.capacity, dim)
        self.filled = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        end = self.filled + key_states.shape[-2]
        self.key_buffer[:, :, self.filled:end] = key_states
        self.value_buffer[:, :, self.filled:end] = value_states
        self.filled = end
        self.keys, self.values = self.key_buffer[:, :, :end], self.value_buffer[:, :, :end]
        return self.keys, self.values


def make_cache(kind, model, capacity):
    if kind == "dynamic":
        return DynamicCache(config=model.config)
    if kind == "dynamic-inplace":
        cache = DynamicCache(config=model.config)
        cache.layers = [InPlaceLayer(capacity) for _ in cache.layers]
        return cache
    return StaticCache(config=model.config, max_cache_len=capacity)


def make_sdpa(filled=None, seen=None):
    # Wraps Transformers' SDPA attention function without modifying library source.
    # filled: ablation for "static-patched". On decode steps, attend over only the
    #   filled cache slots and pass no mask, so Transformers takes its shared-head
    #   (enable_gqa) path instead of expanding K/V from 3 heads to 9.
    # seen: instrumentation. Record what SDPA receives, after any trimming.
    def sdpa(module, query, key, value, attention_mask, **kwargs):
        if filled is not None and query.shape[2] == 1:
            n = filled["n"]
            key, value, attention_mask = key[:, :, :n], value[:, :, :n], None
        if seen is not None and seen["key_len"] is None:
            seen["key_len"] = key.shape[2]
            seen["mask"] = attention_mask is not None
        return ORIGINAL_SDPA(module, query, key, value, attention_mask, **kwargs)
    return sdpa


def set_attention(filled=None, seen=None):
    wrapped = filled is not None or seen is not None
    ALL_ATTENTION_FUNCTIONS["sdpa"] = make_sdpa(filled, seen) if wrapped else ORIGINAL_SDPA


def start(model, kind, prompt, capacity):
    # Prefill into a fresh cache. filled tracks the real cache length for the
    # ablation; the model reads it on each decode step, with no GPU sync needed.
    cache = make_cache(kind, model, capacity)
    filled = {"n": prompt.shape[1]} if kind == "static-patched" else None
    set_attention(filled)
    outputs = model(input_ids=prompt, past_key_values=cache, use_cache=True)
    return outputs.logits[0, -1], outputs.past_key_values, filled


def decode_step(model, logits, cache, filled):
    if filled is not None:
        filled["n"] += 1  # the slot this step writes
    token = logits.argmax().view(1, 1)
    outputs = model(input_ids=token, past_key_values=cache, use_cache=True)
    return outputs.logits[0, -1], outputs.past_key_values


def probe_attention(model, device, logits, cache, filled):
    # Instrumentation, not part of the measurement: run one extra decode step
    # with the attention function and repeat_kv wrapped, to see what SDPA gets.
    seen = {"key_len": None, "mask": None, "expanded": 0}
    original_repeat_kv = sdpa_attention.repeat_kv

    def repeat_kv(hidden_states, n_rep):
        seen["expanded"] += 1
        return original_repeat_kv(hidden_states, n_rep)

    set_attention(filled, seen)
    sdpa_attention.repeat_kv = repeat_kv
    try:
        decode_step(model, logits, cache, filled)
        sync(device)
    finally:
        set_attention(filled)
        sdpa_attention.repeat_kv = original_repeat_kv
    return seen


def measure_length(model, device, kind, length, steps, generator):
    prompt = torch.randint(0, model.config.vocab_size, (1, length), generator=generator).to(device)
    # +1 for the probe step after the timed ones.
    logits, cache, filled = start(model, kind, prompt, length + WARMUP_STEPS + steps + 1)
    del prompt

    for _ in range(WARMUP_STEPS):
        logits, cache = decode_step(model, logits, cache, filled)

    # Count filled slots directly: StaticCache.get_seq_length() reports its capacity here.
    context = length + WARMUP_STEPS
    allocated = sum(kv_bytes_per_layer(cache))
    latencies = []
    for _ in range(steps):
        sync(device)
        begin = time.perf_counter()
        logits, cache = decode_step(model, logits, cache, filled)
        sync(device)
        latencies.append((time.perf_counter() - begin) * 1000)

    seen = probe_attention(model, device, logits, cache, filled)
    set_attention()
    deciles = statistics.quantiles(latencies, n=10)
    del cache
    if device == "mps":
        torch.mps.empty_cache()
    return {
        "cache": kind,
        "context": context,
        "kv_allocated_bytes": allocated,
        "median_ms": statistics.median(latencies),
        "p10_ms": deciles[0],
        "p90_ms": deciles[-1],
        "attention_key_len": seen["key_len"],
        "mask_passed": seen["mask"],
        "kv_expanded": seen["expanded"] > 0,
    }


def check_patch(model, device, generator, length=1024, steps=20):
    # The ablation is only meaningful if it computes the same thing. Generate the
    # same greedy sequence with each cache type and compare against StaticCache.
    prompt = torch.randint(0, model.config.vocab_size, (1, length), generator=generator).to(device)
    runs = {}
    for kind in ("static", "static-patched", "dynamic", "dynamic-inplace"):
        logits, cache, filled = start(model, kind, prompt, length + steps + 1)
        history = [logits.float()]
        for _ in range(steps):
            logits, cache = decode_step(model, logits, cache, filled)
            history.append(logits.float())
        set_attention()
        runs[kind] = torch.stack(history)
    reference = runs["static"]
    for kind in ("static-patched", "dynamic", "dynamic-inplace"):
        same = (runs[kind].argmax(-1) == reference.argmax(-1)).all().item()
        diff = (runs[kind] - reference).abs().max().item()
        print(f"  {kind:<15} vs static: greedy tokens match: {same}, max |logit diff|: {diff:.4f}")


def fit_line(xs, ys):
    # Ordinary least squares, y = intercept + slope * x.
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return intercept, slope, 1 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--cache",
        choices=["dynamic", "dynamic-inplace", "static", "static-patched", "both", "all"],
        default="both",
    )
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    device = args.device
    lengths = [int(n) for n in args.lengths.split(",")]
    kinds = KIND_SETS.get(args.cache, [args.cache])

    print_hardware_info(device)
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("ERROR: MPS is not available on this machine. Re-run with --device cpu.")

    # ---- Memory bandwidth ----------------------------------------------------
    banner("MEMORY BANDWIDTH")
    bandwidth = measure_bandwidth(device)
    print(f"Measured copy bandwidth: {bandwidth:.0f} GB/s (read + write)")
    if device == "mps":
        print(f"Apple's quoted figure:   {APPLE_QUOTED_GBPS} GB/s")
    print()

    # ---- Model ---------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    model.eval()
    config = model.config
    # Measured from real tensors, as in V1: the cache after a one-token call.
    with torch.inference_mode():
        one = model(input_ids=torch.zeros((1, 1), dtype=torch.long, device=device), use_cache=True)
    bytes_per_token = sum(kv_bytes_per_layer(one.past_key_values))
    del one
    trained_limit = config.max_position_embeddings
    print(f"Model: {MODEL_ID}, KV bytes per token: {bytes_per_token}")
    if max(lengths) > trained_limit:
        print(
            f"Note: lengths above {trained_limit} exceed the model's trained context. The generated"
            " text there is meaningless, but the bytes moved per step are real, and bytes are what"
            " this measures."
        )
    print()

    # ---- Sweep ---------------------------------------------------------------
    generator = torch.Generator().manual_seed(0)
    rows = []
    with torch.inference_mode():
        # Unmeasured warmup of both call shapes, as in V1.
        measure_length(model, device, kinds[0], lengths[0], 5, generator)

        if {"static-patched", "dynamic-inplace"} & set(kinds):
            banner("ABLATION CORRECTNESS CHECK")
            check_patch(model, device, generator)
            print()

        for kind in kinds:
            banner(f"DECODE LATENCY, {kind.upper()} CACHE")
            print(f"  {'context':>7}  {'KV MiB':>7}  {'median ms':>9}  {'p10 ms':>6}  {'p90 ms':>6}"
                  f"  {'attn keys':>9}  {'mask':>4}  {'KV expanded':>11}")
            for length in lengths:
                row = measure_length(model, device, kind, length, args.steps, generator)
                rows.append(row)
                print(
                    f"  {row['context']:>7}  {mib(row['context'] * bytes_per_token):>7.1f}"
                    f"  {row['median_ms']:>9.2f}  {row['p10_ms']:>6.2f}  {row['p90_ms']:>6.2f}"
                    f"  {row['attention_key_len']:>9}  {'yes' if row['mask_passed'] else 'no':>4}"
                    f"  {'yes' if row['kv_expanded'] else 'no':>11}"
                )
            print()

        # Thermal check: repeat the first measurement after the machine has been busy.
        first = rows[0]
        again = measure_length(model, device, first["cache"], lengths[0], args.steps, generator)
    drift = (again["median_ms"] - first["median_ms"]) / first["median_ms"] * 100

    # ---- Fit -----------------------------------------------------------------
    banner("LATENCY VS KV BYTES (straight-line fit)")
    for kind in kinds:
        points = [r for r in rows if r["cache"] == kind]
        xs = [r["context"] * bytes_per_token for r in points]
        ys = [r["median_ms"] for r in points]
        intercept, slope, r2 = fit_line(xs, ys)
        effective_gbps = 1 / slope / 1e6  # slope is ms per byte
        print(f"{kind}:")
        print(f"  empty-cache cost (intercept): {intercept:.2f} ms")
        print(f"  effective bandwidth (slope):  {effective_gbps:.0f} GB/s of KV per extra ms")
        print(f"  passes over the cache/step:   {bandwidth / effective_gbps:.1f}  (measured bandwidth / effective)")
        print(f"  fit R^2:                      {r2:.4f}")
    print()

    banner("THERMAL CHECK")
    print(f"First {first['cache']} run at {first['context']} tokens: {first['median_ms']:.2f} ms")
    print(f"Same measurement at the end:  {again['median_ms']:.2f} ms  ({drift:+.1f}%)")
    if abs(drift) > 5:
        print("WARNING: more than 5% drift; the machine may have throttled during the run.")
    print()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) + ["kv_bytes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "kv_bytes": row["context"] * bytes_per_token})
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
