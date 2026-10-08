"""The kernels in one decode step, eager and as a CUDA graph (SPEC.md V13).

    modal run kv_kernels.py

One H100. Qwen3-8B in BF16 with Hugging Face transformers and a static KV cache, batch 1,
a 256-token prompt. For each of two runs:
  eager:  decode steps launched op by op from Python, timed per step
  graph:  the same step captured once with torch.cuda.graph and replayed, timed per step
  both:   one profiled step at a time (torch.profiler) for kernel counts, kernel time and the
          kernel names
Greedy tokens from the eager and graph loops must match. Writes results_kernels.csv,
results_kernels_groups.csv and results_kernels_notes.txt from the local client, so run it
without --detach.
"""
import pathlib
import re

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "transformers", "accelerate")
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
app = modal.App("kv-kernels")
MODEL = "Qwen/Qwen3-8B"
OUT = pathlib.Path(__file__).parent

PROMPT_TOKENS = 256
STEPS = 20          # timed decode steps per case
PROFILED = 5        # profiled steps per case, one profile each
WARMUP = 5
REPEATS = 2
MAX_LEN = PROMPT_TOKENS + WARMUP + 2 * STEPS + 2 * PROFILED + 16

# Kernel families, matched on the kernel name in order. The rest is "other".
FAMILIES = [
    ("matrix multiply", r"gemm|gemv|cutlass|nvjet|xmma|cublas|splitK"),
    ("attention", r"flash|fmha|attention|sdpa|efficient"),
    ("reduction", r"reduce|softmax|mean|norm|argmax"),
    ("copy or concat", r"copy|cat|index|scatter|gather|fill"),
    ("elementwise", r"elementwise|vectorized|unrolled"),
]


def family(name: str) -> str:
    for fam, pat in FAMILIES:
        if re.search(pat, name, re.I):
            return fam
    return "other"


@app.function(gpu="H100!", cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def run() -> dict:
    import os
    os.environ["HF_HOME"] = "/hf"
    import subprocess
    import time

    import torch
    import transformers
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

    notes = []
    note = notes.append
    note("gpu: " + subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip())
    note(f"torch {torch.__version__}, transformers {transformers.__version__}, cuda {torch.version.cuda}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    note(f"model {MODEL}, attention {model.config._attn_implementation}, layers {model.config.num_hidden_layers}")
    text = "The capital of France is Paris. Paris is a major city in France, known for its museums. " * 40
    prompt = tok(text, return_tensors="pt").input_ids[:, :PROMPT_TOKENS].cuda()

    def new_cache():
        try:
            return StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_LEN, device="cuda", dtype=torch.bfloat16)
        except TypeError:  # newer transformers dropped the batch, device and dtype arguments
            return StaticCache(config=model.config, max_cache_len=MAX_LEN)

    # Static inputs for one decode step. The 4D mask is built here and updated in place, so the
    # model never builds one itself (that path can sync with the CPU, which a graph can't hold).
    ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.zeros(1, dtype=torch.long, device="cuda")
    mask = torch.full((1, 1, 1, MAX_LEN), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device="cuda")
    state = {}

    # One cache for the whole run: a captured graph keeps pointing at the cache's memory, so
    # prefill resets this cache in place instead of making a new one.
    cache = new_cache()

    def prefill():
        cache.reset()
        n = prompt.shape[1]
        cp = torch.arange(n, device="cuda")
        with torch.no_grad():
            logits = model(input_ids=prompt, past_key_values=cache, cache_position=cp, position_ids=cp[None], use_cache=True).logits
        state["n"] = n
        mask.fill_(torch.finfo(torch.bfloat16).min)
        mask[..., :n].zero_()
        return logits[:, -1].argmax(-1)

    def step():
        with torch.no_grad():
            logits = model(input_ids=ids, past_key_values=cache, cache_position=pos, position_ids=pos[None],
                           attention_mask=mask, use_cache=True).logits
        return logits[:, -1].argmax(-1)

    def advance(next_token):
        # Feed the picked token back and open one more cache slot: tiny copies outside the step.
        ids.copy_(next_token.view(1, 1))
        pos.fill_(state["n"])
        mask[..., state["n"]].zero_()
        state["n"] += 1

    def timed(run_step, label, rep, rows, tokens):
        for _ in range(WARMUP):
            advance(run_step())
        times = []
        for _ in range(STEPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run_step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            tokens.append(int(out))
            advance(out)
        times.sort()
        wall = times[len(times) // 2]
        kernels, busy, span, names = profiled(run_step)
        rows.append(dict(run=rep, case=label, wall_ms=round(wall, 3), tokens_per_s=round(1000 / wall, 1),
                         kernels_per_step=kernels, kernel_ms=round(busy, 3), first_to_last_kernel_ms=round(span, 3),
                         idle_ms=round(wall - busy, 3)))
        return names

    def profiled(run_step):
        counts, busies, spans, names = [], [], [], {}
        for i in range(PROFILED):
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                out = run_step()
                torch.cuda.synchronize()
            ks = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and e.name != "cudaDeviceSynchronize"),
                        key=lambda e: e.time_range.start)
            spans_us = [(e.time_range.start, e.time_range.end) for e in ks]
            merged = 0.0  # union of kernel intervals, so overlaps aren't counted twice
            cur_s, cur_e = None, None
            for s, e in spans_us:
                if cur_e is None or s > cur_e:
                    if cur_e is not None:
                        merged += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            if cur_e is not None:
                merged += cur_e - cur_s
            counts.append(len(ks))
            busies.append(merged / 1000)
            spans.append((spans_us[-1][1] - spans_us[0][0]) / 1000 if ks else 0.0)
            if i == PROFILED - 1:
                for e in ks:
                    n = names.setdefault(e.name, [0, 0.0])
                    n[0] += 1
                    n[1] += (e.time_range.end - e.time_range.start) / 1000
            advance(out)
        mid = len(counts) // 2
        return sorted(counts)[mid], sorted(busies)[mid], sorted(spans)[mid], names

    rows, groups = [], []
    for rep in range(1, REPEATS + 1):
        eager_tokens, graph_tokens = [], []
        advance(prefill())
        names = timed(step, "eager", rep, rows, eager_tokens)
        for n, (c, ms) in names.items():
            groups.append(dict(run=rep, case="eager", family=family(n), kernel=n[:160], count=c, ms=round(ms, 4)))

        # The same step as a CUDA graph: capture once on a side stream, then replay.
        advance(prefill())
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                graph_out = step()

            def replay():
                g.replay()
                return graph_out.clone()

            # Restart from a fresh prefill so the graph decodes the same tokens as eager did.
            advance(prefill())
            names = timed(replay, "cuda graph", rep, rows, graph_tokens)
            for n, (c, ms) in names.items():
                groups.append(dict(run=rep, case="cuda graph", family=family(n), kernel=n[:160], count=c, ms=round(ms, 4)))
            same = eager_tokens == graph_tokens
            note(f"run {rep}: graph tokens match eager: {same} ({len(eager_tokens)} tokens)")
            del g
        except Exception as e:  # report and keep the eager results
            note(f"run {rep}: CUDA graph capture failed: {type(e).__name__}: {str(e)[:400]}")
    note("decoded text (run 1, eager): " + repr(tok.decode(eager_tokens)[:200]))
    return dict(rows=rows, groups=groups, notes=notes)


@app.local_entrypoint()
def main():
    import csv
    res = run.remote()
    for name, key in [("results_kernels.csv", "rows"), ("results_kernels_groups.csv", "groups")]:
        rows = res[key]
        if not rows:
            continue
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    open(OUT / "results_kernels_notes.txt", "w").write("\n".join(res["notes"]))
    print("\n".join(res["notes"]))
    for r in res["rows"]:
        print(r)
