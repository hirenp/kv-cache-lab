"""KV Cache Lab V3: moving a KV cache vs rebuilding it.

Reads top to bottom: for each model and context length, time a prefill
(recompute), then copy the resulting cache out to each storage tier and back
(offload / reload), check the reloaded bytes decode identically, and find the
context length where reloading beats recomputing.
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from kv_bandwidth import measure_bandwidth, sync
from kv_lab import banner, mib

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
REPEATS = 5
CHECK_STEPS = 20
# Nominal link speeds in GB/s, for the labelled network prediction only.
LINKS = {"100 Gb/s network": 12.5, "400 Gb/s network": 50.0}


def timed(fn, setup=None):
    # One warmup, then the median of REPEATS runs, each bracketed by a sync.
    # setup runs before each measured run, outside the timed region.
    if setup:
        setup()
    fn()
    times = []
    for _ in range(REPEATS):
        if setup:
            setup()
        sync("cuda")
        start = time.perf_counter()
        fn()
        sync("cuda")
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def cache_tensors(cache):
    return [t for layer in cache.layers for t in (layer.keys, layer.values)]


def cache_from(model, tensors):
    # Build a fresh DynamicCache from a flat [K0, V0, K1, V1, ...] list.
    cache = DynamicCache(config=model.config)
    for i in range(0, len(tensors), 2):
        cache.update(tensors[i], tensors[i + 1], i // 2)
    return cache


def greedy_decode(model, first_logits, cache, steps):
    # cuDNN attention has a large cost per new shape on H100 (see SPEC V3.6).
    torch.backends.cuda.enable_cudnn_sdp(False)
    logits, tokens, history = first_logits, [], []
    for _ in range(steps):
        token = logits.argmax().view(1, 1)
        tokens.append(token.item())
        outputs = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache, logits = outputs.past_key_values, outputs.logits[0, -1]
        history.append(logits.float())
    torch.backends.cuda.enable_cudnn_sdp(True)
    return tokens, torch.stack(history)


def as_bytes(tensor):
    # bf16 has no numpy type; view the same memory as raw bytes.
    return tensor.view(torch.uint8).numpy()


class FileTier:
    # A cache written to one file in dir_path. Reload reads it into pinned memory
    # and copies to the GPU. cold drops the file from the OS page cache first.
    def __init__(self, dir_path, gpu_tensors):
        self.path = os.path.join(dir_path, "kv_cache.bin")
        self.src = gpu_tensors
        self.pinned = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in gpu_tensors]
        self.back = [torch.empty_like(t) for t in gpu_tensors]

    def offload(self):
        for p, s in zip(self.pinned, self.src):
            p.copy_(s)
        with open(self.path, "wb") as f:
            for p in self.pinned:
                f.write(as_bytes(p))
            f.flush()
            os.fsync(f.fileno())

    def drop_page_cache(self):
        fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    def reload(self):
        with open(self.path, "rb", buffering=0) as f:
            for p in self.pinned:
                f.readinto(as_bytes(p))
        for b, p in zip(self.back, self.pinned):
            b.copy_(p, non_blocking=True)

    def remove(self):
        os.remove(self.path)


def memory_tier(gpu_tensors, where):
    # GPU buffer, pinned host memory or pageable host memory.
    if where == "gpu":
        store = [torch.empty_like(t) for t in gpu_tensors]
    else:
        store = [torch.empty(t.shape, dtype=t.dtype, pin_memory=(where == "pinned")) for t in gpu_tensors]
    back = [torch.empty_like(t) for t in gpu_tensors]

    def offload():
        for d, s in zip(store, gpu_tensors):
            d.copy_(s, non_blocking=(where == "pinned"))

    def reload():
        for b, d in zip(back, store):
            b.copy_(d, non_blocking=(where == "pinned"))

    return offload, reload, back


def print_gpu_link():
    if shutil.which("nvidia-smi") is None:
        return
    query = "name,pcie.link.gen.max,pcie.link.width.max,memory.total"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True
    )
    print(f"nvidia-smi: {result.stdout.strip()}  ({query})")


def run_model(model_key, lengths, dirs, generator):
    model_id = MODELS[model_key]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    limit = model.config.max_position_embeddings
    usable = [n for n in lengths if n <= limit]
    banner(f"{model_id}")
    if len(usable) < len(lengths):
        print(f"Skipping lengths above the trained context ({limit}): {[n for n in lengths if n > limit]}")

    rows = []
    with torch.inference_mode():
        one = model(input_ids=torch.zeros((1, 1), dtype=torch.long, device="cuda"), use_cache=True)
        bytes_per_token = sum(t.numel() * t.element_size() for t in cache_tensors(one.past_key_values))
        del one
        print(f"KV bytes per token (measured): {bytes_per_token}")
        tiers = ["gpu", "pinned", "pageable"] + list(dirs)
        print(f"  {'context':>7}  {'KV MiB':>8}  {'recompute':>9}  " + "  ".join(f"{t:>14}" for t in tiers))
        print(f"  {'':>7}  {'':>8}  {'ms':>9}  " + "  ".join(f"{'off / reload':>14}" for _ in tiers))

        for length in usable:
            prompt = torch.randint(0, model.config.vocab_size, (1, length), generator=generator).to("cuda")

            def prefill():
                return model(input_ids=prompt, use_cache=True, logits_to_keep=1)

            recompute_ms = timed(prefill)
            outputs = prefill()
            original = cache_tensors(outputs.past_key_values)
            first_logits = outputs.logits[0, -1]
            kv_bytes = sum(t.numel() * t.element_size() for t in original)
            reference_tokens, reference_logits = greedy_decode(
                model, first_logits, cache_from(model, original), CHECK_STEPS
            )

            row = {"model": model_key, "context": length, "kv_bytes": kv_bytes, "recompute_ms": recompute_ms}
            cells = []
            for tier in tiers:
                if tier in dirs:
                    file_tier = FileTier(dirs[tier], original)
                    offload_ms = timed(file_tier.offload)
                    reload_ms = timed(file_tier.reload, setup=file_tier.drop_page_cache)
                    row[f"{tier}_warm_reload_ms"] = timed(file_tier.reload)
                    back = file_tier.back
                else:
                    offload, reload, back = memory_tier(original, tier)
                    offload_ms = timed(offload)
                    reload_ms = timed(reload)

                same_bytes = all(torch.equal(a, b) for a, b in zip(original, back))
                tokens, logits = greedy_decode(model, first_logits, cache_from(model, back), CHECK_STEPS)
                same_decode = tokens == reference_tokens and torch.equal(logits, reference_logits)
                if tier in dirs:
                    file_tier.remove()
                row[f"{tier}_offload_ms"] = offload_ms
                row[f"{tier}_reload_ms"] = reload_ms
                row[f"{tier}_identical"] = same_bytes and same_decode
                cells.append(f"{offload_ms:>6.1f} / {reload_ms:>5.1f}{'' if same_bytes and same_decode else '!'}")
                del back
            print(f"  {length:>7}  {mib(kv_bytes):>8.1f}  {recompute_ms:>9.1f}  " + "  ".join(f"{c:>14}" for c in cells))
            rows.append(row)
            del outputs, original
            torch.cuda.empty_cache()

    print("  (disk reloads are cold: the file is dropped from the page cache before each read;")
    print("   warm reloads are in the CSV. '!' would mark a reload that did not decode identically.)")
    print()
    report_breakeven(rows, tiers)
    return rows


def report_breakeven(rows, tiers):
    print("Where reloading beats recomputing (tested contexts):")
    for tier in tiers:
        faster = [r["context"] for r in rows if r[f"{tier}_reload_ms"] < r["recompute_ms"]]
        largest = rows[-1]
        ratio = largest["recompute_ms"] / largest[f"{tier}_reload_ms"]
        verdict = ", ".join(str(n) for n in faster) if faster else "none"
        print(f"  {tier:<9} reload faster at: {verdict:<30} recompute/reload at {largest['context']}: {ratio:.1f}x")
    print()
    print("Prediction, not measured: time to move the cache over a network link (bytes / nominal speed):")
    for r in rows:
        predictions = "  ".join(f"{name}: {r['kv_bytes'] / (gbps * 1e9) * 1000:.1f} ms" for name, gbps in LINKS.items())
        print(f"  {r['context']:>7} tokens  recompute {r['recompute_ms']:.1f} ms   {predictions}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=list(MODELS) + ["both"], default="both")
    parser.add_argument("--lengths", default="1024,4096,16384,32768")
    parser.add_argument("--disk-dir", default=tempfile.gettempdir(), help="local disk tier")
    parser.add_argument("--net-dir", default=None, help="optional network storage tier (e.g. a mounted volume)")
    parser.add_argument("--out", default="results_offload.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: kv_offload.py needs a CUDA GPU.")

    lengths = [int(n) for n in args.lengths.split(",")]
    models = list(MODELS) if args.model == "both" else [args.model]
    dirs = {"disk": args.disk_dir}
    if args.net_dir:
        dirs["net"] = args.net_dir

    banner("GPU")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print_gpu_link()
    print(f"Measured GPU copy bandwidth: {measure_bandwidth('cuda'):.0f} GB/s (read + write)")
    print()

    generator = torch.Generator().manual_seed(0)
    rows = []
    for model_key in models:
        rows += run_model(model_key, lengths, dirs, generator)
        torch.cuda.empty_cache()  # the previous model is freed once run_model returns

    first = ["model", "context", "kv_bytes", "recompute_ms"]
    fields = first + sorted({k for r in rows for k in r} - set(first))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
