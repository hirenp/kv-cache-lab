"""KV Cache Lab V5: what a long prefill does to someone who is already decoding.

Reads top to bottom: user A is decoding; request B arrives with a long prompt.
Run B's prefill on A's GPU all at once (colocated), in chunks interleaved with
A's decode steps (chunked), or on the other GPU in parallel (disaggregated),
and record every gap between A's tokens plus B's time to first token.
"""

import argparse
import csv
import statistics
import threading
import time

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, DynamicCache

from kv_disagg import DECODE_GPU, MODEL_ID, PREFILL_GPU
from kv_lab import banner
from kv_offload import cache_tensors

A_PROMPT = 1024
A_STEPS = 100
ARRIVAL = 20  # B arrives just before A's 20th decode step
CHUNKS = [512, 2048]


def sync(device):
    torch.cuda.synchronize(device)


def prefill(model, ids, cache=None):
    cache = cache if cache is not None else DynamicCache(config=model.config)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return out.logits[0, -1], out.past_key_values


def run(mode, models, a_prompt, b_prompt, worker=None):
    # Returns A's gaps between tokens (ms), A's tokens, B's time to first token (ms)
    # and B's first token. Every gap ends with a sync of A's GPU, so it includes all
    # work the scheduler ran on that GPU in between.
    a_device = DECODE_GPU if mode.startswith("disaggregated") else PREFILL_GPU
    a_model = models[a_device]
    logits, cache = prefill(a_model, a_prompt.to(a_device))
    sync(a_device)

    b = {"ttft_ms": None, "first_token": None}
    chunk = int(mode.split("-")[1]) if mode.startswith("chunked") else None
    b_cache, b_pos = None, 0
    thread = None
    gaps, tokens = [], []
    previous = time.perf_counter()

    for step in range(A_STEPS):
        if b_prompt is not None and step == ARRIVAL:
            arrival = previous
            if mode == "colocated":
                b_logits, _ = prefill(models[PREFILL_GPU], b_prompt)
                sync(PREFILL_GPU)
                b["ttft_ms"] = (time.perf_counter() - arrival) * 1000
                b["first_token"] = b_logits.argmax().item()
            elif mode == "disaggregated-thread":
                thread = threading.Thread(target=prefill_elsewhere, args=(models, b_prompt, arrival, b))
                thread.start()
            elif mode == "disaggregated-process":
                worker[0].put(b_prompt.cpu())
            else:
                b_cache = DynamicCache(config=models[PREFILL_GPU].config)
        if chunk and b_cache is not None and b_pos < b_prompt.shape[1]:
            b_logits, b_cache = prefill(models[PREFILL_GPU], b_prompt[:, b_pos:b_pos + chunk], b_cache)
            b_pos += chunk
            if b_pos >= b_prompt.shape[1]:
                sync(PREFILL_GPU)
                b["ttft_ms"] = (time.perf_counter() - arrival) * 1000
                b["first_token"] = b_logits.argmax().item()

        token = logits.argmax().view(1, 1)
        out = a_model(input_ids=token, past_key_values=cache, use_cache=True)
        logits, cache = out.logits[0, -1], out.past_key_values
        sync(a_device)
        now = time.perf_counter()
        gaps.append((now - previous) * 1000)
        tokens.append(token.item())
        previous = now

    if thread:
        thread.join()
    if mode == "disaggregated-process" and b_prompt is not None:
        # perf_counter is CLOCK_MONOTONIC on Linux, so the worker's timestamp is comparable.
        finished, b["first_token"] = worker[1].get()
        b["ttft_ms"] = (finished - arrival) * 1000
    return gaps, tokens, b


def prefill_worker(requests, results):
    # "disaggregated-process": B's prefill in its own process with its own copy of
    # the model on GPU 0, the way disaggregated systems run separate prefill workers.
    # The cache stays on GPU 0; moving it is part 2's measurement (about 6 ms at 32k).
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(PREFILL_GPU).eval()
    with torch.inference_mode():
        while (prompt := requests.get()) is not None:
            logits, _ = prefill(model, prompt.to(PREFILL_GPU))
            sync(PREFILL_GPU)
            results.put((time.perf_counter(), logits.argmax().item()))


def prefill_elsewhere(models, b_prompt, arrival, b):
    # "disaggregated-thread": B's prefill on GPU 0 in a thread of the same process,
    # then its cache copied to GPU 1. A's loop on GPU 1 doesn't wait for it on the
    # GPU, but both threads share one Python interpreter lock.
    # inference_mode is per thread, so it has to be entered again here.
    with torch.inference_mode():
        b_logits, b_cache = prefill(models[PREFILL_GPU], b_prompt)
        dst = [t.to(DECODE_GPU) for t in cache_tensors(b_cache)]
        sync(PREFILL_GPU)
        sync(DECODE_GPU)
    b["ttft_ms"] = (time.perf_counter() - arrival) * 1000
    b["first_token"] = b_logits.argmax().item()
    del dst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lengths", default="4096,16384,32768", help="B's prompt lengths")
    parser.add_argument("--out", default="results_interference.csv")
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit("ERROR: kv_interference.py needs two CUDA GPUs.")
    lengths = [int(n) for n in args.lengths.split(",")]
    # cuDNN attention has a large cost per new shape on H100 (SPEC V3.6), and a global
    # setting can't be flipped per call while B's prefill runs in another thread.
    torch.backends.cuda.enable_cudnn_sdp(False)

    models = {d: AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(d).eval()
              for d in (PREFILL_GPU, DECODE_GPU)}
    generator = torch.Generator().manual_seed(0)
    vocab = models[PREFILL_GPU].config.vocab_size
    a_prompt = torch.randint(0, vocab, (1, A_PROMPT), generator=generator)
    modes = ["colocated"] + [f"chunked-{c}" for c in CHUNKS] + ["disaggregated-thread", "disaggregated-process"]

    context = mp.get_context("spawn")
    worker = (context.Queue(), context.Queue())
    process = context.Process(target=prefill_worker, args=worker)
    process.start()
    rows, summaries = [], []

    with torch.inference_mode():
        # Warm up both GPUs on A's shapes and B's longest prompt, unmeasured.
        for device, model in models.items():
            prefill(model, torch.randint(0, vocab, (1, max(lengths)), generator=generator).to(device))
        run("disaggregated-thread", models, a_prompt, None)
        for n in (max(lengths), 4096):
            worker[0].put(torch.randint(0, vocab, (1, n), generator=generator))
            worker[1].get()
        warm_b = torch.randint(0, vocab, (1, 4096), generator=generator).to(PREFILL_GPU)
        for c in CHUNKS:
            run(f"chunked-{c}", models, a_prompt, warm_b)

        alone, reference_tokens, _ = run("colocated", models, a_prompt, None)
        summaries.append(("alone", "-", alone, None, True, None))
        rows += [{"mode": "alone", "b_tokens": 0, "step": i, "gap_ms": g} for i, g in enumerate(alone)]

        for length in lengths:
            b_prompt = torch.randint(0, vocab, (1, length), generator=generator).to(PREFILL_GPU)
            reference_b = prefill(models[PREFILL_GPU], b_prompt)[0].argmax().item()
            for mode in modes:
                gaps, tokens, b = run(mode, models, a_prompt, b_prompt, worker)
                summaries.append((mode, length, gaps, b, tokens == reference_tokens, b["first_token"] == reference_b))
                rows += [{"mode": mode, "b_tokens": length, "step": i, "gap_ms": g} for i, g in enumerate(gaps)]
            torch.cuda.empty_cache()
    worker[0].put(None)
    process.join()

    banner("USER A'S GAPS BETWEEN TOKENS, AND REQUEST B'S TIME TO FIRST TOKEN")
    print(f"  {'mode':<21} {'B tokens':>8}  {'A median':>8}  {'A worst':>8}  {'A gaps > 2x':>11}  {'B TTFT':>8}"
          f"  {'A same':>6}  {'B same':>6}")
    baseline = statistics.median(alone)
    for mode, length, gaps, b, a_same, b_same in summaries:
        slow = sum(g > 2 * baseline for g in gaps)
        ttft = f"{b['ttft_ms']:.0f} ms" if b else "-"
        print(f"  {mode:<21} {length:>8}  {statistics.median(gaps):>6.1f}ms  {max(gaps):>6.0f}ms  {slow:>11}  {ttft:>8}"
              f"  {str(a_same):>6}  {str(b_same) if b else '-':>6}")
    print(f"  ('A gaps > 2x' counts gaps over twice A's median gap when alone, {baseline:.1f} ms. 'A same': A's")
    print("   tokens match A decoding alone. 'B same': B's first token matches one unchunked prefill.)")
    print()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "b_tokens", "step", "gap_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
