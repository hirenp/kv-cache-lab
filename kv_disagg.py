"""KV Cache Lab V4: prefill on one GPU, decode on another.

Reads top to bottom: check how the two GPUs are connected and how fast, then
for each context length time three ways of getting to the first decode step:
colocated on GPU 0, prefill on GPU 0 with a bulk cache copy to GPU 1, and the
same with the cache copied layer by layer while prefill runs.
"""

import argparse
import csv
import shutil
import statistics
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from kv_lab import banner, mib
from kv_offload import LINKS, cache_tensors

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
REPEATS = 9
CHECK_STEPS = 20
PREFILL_GPU, DECODE_GPU = "cuda:0", "cuda:1"


def sync():
    torch.cuda.synchronize(PREFILL_GPU)
    torch.cuda.synchronize(DECODE_GPU)


def timed(fn):
    # One warmup, then REPEATS runs, each bracketed by a sync of both GPUs.
    # Returns (median, max - min) so noisy differences are visible.
    fn()
    times = []
    for _ in range(REPEATS):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), max(times) - min(times)


def print_topology():
    # nvidia-smi topo -m does not work inside Modal's container; nvlink -s does.
    if shutil.which("nvidia-smi"):
        for args in (["topo", "-m"], ["nvlink", "-s", "-i", "0"]):
            result = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
            print(f"$ nvidia-smi {' '.join(args)}")
            print((result.stdout or result.stderr).strip() or "(no output)")
    print(f"Peer access GPU 0 -> GPU 1: {torch.cuda.can_device_access_peer(0, 1)}")


def link_bandwidth():
    x = torch.empty(2**29, dtype=torch.bfloat16, device=PREFILL_GPU)  # 1 GiB
    y = torch.empty(2**29, dtype=torch.bfloat16, device=DECODE_GPU)
    for _ in range(3):
        y.copy_(x)
    sync()
    start = time.perf_counter()
    for _ in range(20):
        y.copy_(x)
    sync()
    seconds = (time.perf_counter() - start) / 20
    return x.numel() * x.element_size() / seconds / 1e9


def cache_on(model, tensors):
    # Wrap already-placed K/V tensors in a DynamicCache without copying them.
    cache = DynamicCache(config=model.config)
    for layer, i in zip(cache.layers, range(0, len(tensors), 2)):
        layer.lazy_initialization(tensors[i], tensors[i + 1])
        layer.keys, layer.values = tensors[i], tensors[i + 1]
    return cache


def decode(model, logits, cache, steps):
    # cuDNN attention has a large cost per new shape on H100 (see SPEC V3.6).
    torch.backends.cuda.enable_cudnn_sdp(False)
    tokens, history = [], []
    for _ in range(steps):
        token = logits.argmax().view(1, 1).to(model.device)
        outputs = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache, logits = outputs.past_key_values, outputs.logits[0, -1]
        tokens.append(token.item())
        history.append(logits.float().cpu())
    torch.backends.cuda.enable_cudnn_sdp(True)
    return tokens, torch.stack(history)


class LayerCopier:
    # Forward hooks on each decoder layer. When a layer finishes during prefill,
    # its K/V are already in the cache, so start copying them to the decode GPU
    # on a side stream while the next layer computes. copy=False keeps the hook
    # and stream bookkeeping but skips the copy, to measure that overhead alone.
    def __init__(self, model):
        self.stream = torch.cuda.Stream(device=PREFILL_GPU)
        self.cache, self.dst, self.copy = None, None, False
        self.handles = [layer.register_forward_hook(self.hook(i)) for i, layer in enumerate(model.model.layers)]

    def hook(self, i):
        def fn(module, args, output):
            if self.dst is None:
                return
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(PREFILL_GPU))
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                if self.copy:
                    layer = self.cache.layers[i]
                    self.dst[2 * i].copy_(layer.keys, non_blocking=True)
                    self.dst[2 * i + 1].copy_(layer.values, non_blocking=True)
        return fn

    def remove(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lengths", default="1024,4096,16384,32768")
    parser.add_argument("--out", default="results_disagg.csv")
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit("ERROR: kv_disagg.py needs two CUDA GPUs.")
    lengths = [int(n) for n in args.lengths.split(",")]

    banner("GPUS AND LINK")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}   GPU 1: {torch.cuda.get_device_name(1)}")
    print_topology()
    bandwidth = link_bandwidth()
    print(f"Measured GPU 0 -> GPU 1 copy bandwidth: {bandwidth:.0f} GB/s")
    if bandwidth < 100:
        raise SystemExit("ERROR: under 100 GB/s, so the GPUs are not on NVLink. Stopping (see SPEC V4.1).")
    print()

    prefill_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(PREFILL_GPU).eval()
    decode_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DECODE_GPU).eval()
    copier = LayerCopier(prefill_model)
    generator = torch.Generator().manual_seed(0)
    rows = []

    banner(f"{MODEL_ID}: PREFILL ON GPU 0, DECODE ON GPU 0 OR GPU 1")
    print(f"  {'context':>7}  {'KV MiB':>7}  {'prefill':>8}  {'spread':>6}  {'copy only':>13}  {'layer-wise':>13}"
          f"  {'hooks only':>10}  {'1st decode':>10}  {'1st decode':>10}")
    print(f"  {'':>7}  {'':>7}  {'ms':>8}  {'ms':>6}  {'ms (%)':>13}  {'+ms (%)':>13}  {'+ms':>10}  {'GPU 0 ms':>10}  {'GPU 1 ms':>10}")

    with torch.inference_mode():
        for length in lengths:
            prompt = torch.randint(0, prefill_model.config.vocab_size, (1, length), generator=generator).to(PREFILL_GPU)
            state = {}

            def prefill():
                copier.cache = DynamicCache(config=prefill_model.config)
                out = prefill_model(input_ids=prompt, past_key_values=copier.cache, use_cache=True, logits_to_keep=1)
                state["logits"], state["cache"] = out.logits[0, -1], out.past_key_values

            prefill()
            original = cache_tensors(state["cache"])
            dst = [torch.empty(t.shape, dtype=t.dtype, device=DECODE_GPU) for t in original]
            kv_bytes = sum(t.numel() * t.element_size() for t in original)

            copier.dst = None
            prefill_ms, prefill_spread_ms = timed(prefill)

            # Both transfers end with a sync, so "done" means every byte has landed on GPU 1.
            def bulk():
                prefill()
                for d, s in zip(dst, cache_tensors(state["cache"])):
                    d.copy_(s)
                sync()
            bulk_ms = timed(bulk)[0]

            # The transfer on its own, from a cache that is already built. Subtracting
            # two prefill medians is too noisy for a few milliseconds of copying.
            def copy_only():
                for d, s in zip(dst, cache_tensors(state["cache"])):
                    d.copy_(s)
                sync()
            copy_ms = timed(copy_only)[0]

            def layerwise(copy):
                def run():
                    copier.dst, copier.copy = dst, copy
                    prefill()
                    copier.dst = None
                    sync()
                return run
            layerwise_ms = timed(layerwise(True))[0]
            hooks_only_ms = timed(layerwise(False))[0]

            # First decode step on each GPU, from a cache already in place.
            first_gpu0_ms = timed(lambda: decode(prefill_model, state["logits"], cache_on(prefill_model, cache_tensors(state["cache"])), 1))[0]
            layerwise(True)()
            first_gpu1_ms = timed(lambda: decode(decode_model, state["logits"], cache_on(decode_model, dst), 1))[0]

            # Correctness: the cache that arrived on GPU 1 (layer-wise, then bulk)
            # must decode the same as colocated on GPU 0.
            reference = decode(prefill_model, state["logits"], cache_on(prefill_model, cache_tensors(state["cache"])), CHECK_STEPS)
            layerwise(True)()
            via_layers = decode(decode_model, state["logits"], cache_on(decode_model, dst), CHECK_STEPS)
            bulk()
            via_bulk = decode(decode_model, state["logits"], cache_on(decode_model, dst), CHECK_STEPS)
            checks = {}
            for name, (tokens, logits) in (("layerwise", via_layers), ("bulk", via_bulk)):
                checks[f"{name}_tokens_match"] = tokens == reference[0]
                checks[f"{name}_max_logit_diff"] = (logits - reference[1]).abs().max().item()

            row = {
                "context": length, "kv_bytes": kv_bytes, "prefill_ms": prefill_ms, "prefill_spread_ms": prefill_spread_ms,
                "copy_only_ms": copy_ms,
                "bulk_extra_ms": bulk_ms - prefill_ms, "layerwise_extra_ms": layerwise_ms - prefill_ms,
                "hooks_only_extra_ms": hooks_only_ms - prefill_ms,
                "first_decode_gpu0_ms": first_gpu0_ms, "first_decode_gpu1_ms": first_gpu1_ms, **checks,
            }
            rows.append(row)
            def pct(ms):
                return f"{ms:+.1f} ({ms / prefill_ms * 100:+.1f}%)"
            print(f"  {length:>7}  {mib(kv_bytes):>7.0f}  {prefill_ms:>8.1f}  {prefill_spread_ms:>6.1f}  {pct(copy_ms):>13}"
                  f"  {pct(row['layerwise_extra_ms']):>13}  {row['hooks_only_extra_ms']:>+10.1f}"
                  f"  {first_gpu0_ms:>10.1f}  {first_gpu1_ms:>10.1f}")
            torch.cuda.empty_cache()
    copier.remove()
    print()

    banner("CORRECTNESS (20 decoded tokens vs colocated on GPU 0)")
    for r in rows:
        print(f"  {r['context']:>7}: layer-wise tokens match {r['layerwise_tokens_match']}, max |logit diff| "
              f"{r['layerwise_max_logit_diff']:.4f};  bulk tokens match {r['bulk_tokens_match']}, "
              f"max |logit diff| {r['bulk_max_logit_diff']:.4f}")
    print()

    banner("FOR COMPARISON: PART 3'S NETWORK PREDICTION (bytes / nominal speed, not measured)")
    for r in rows:
        predicted = "  ".join(f"{name}: {r['kv_bytes'] / (gbps * 1e9) * 1000:.1f} ms" for name, gbps in LINKS.items())
        print(f"  {r['context']:>7} tokens  NVLink copy (measured): {r['copy_only_ms']:.1f} ms   {predicted}")
    print()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
