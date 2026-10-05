"""One model in BF16, FP8 and FP4 on a B200 and an H100: memory, speed, accuracy (SPEC.md V12).

    modal run kv_formats.py

Runs the B200 and H100 containers at the same time. In each, one `vllm serve` at a time, with
`vllm bench serve` and a GSM8K check against it from the same container. For each config:
  memory:   weight memory and KV cache capacity from vLLM's startup log
  decode:   mean time per output token, 1 user (256-token prompts) and 16 users (8,192-token prompts)
  prefill:  mean time to first token for one 8,192-token prompt at a time
  accuracy: the 1,319 GSM8K test questions, greedy, thinking off (B200 only)
Each speed run twice. Writes results_formats.csv and results_formats_notes.txt from the local
client, so run it without --detach.
"""
import pathlib
import time

import modal

VLLM_VERSION = "0.30.0"
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "datasets")
)
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
app = modal.App("kv-formats")
OUT = pathlib.Path(__file__).parent

# (label, model, extra `vllm serve` args)
WEIGHTS = [
    ("bf16", "Qwen/Qwen3-8B", []),
    ("fp8", "Qwen/Qwen3-8B-FP8", []),
    ("fp4", "RedHatAI/Qwen3-8B-NVFP4", []),
]
KV_FP8 = ("bf16-kv-fp8", "Qwen/Qwen3-8B", ["--kv-cache-dtype", "fp8"])
SEED = 0
REPEATS = 2
# (case, users, input tokens, output tokens, prompts)
SPEED = [("decode, 1 user", 1, 256, 512, 8), ("prefill 8k", 1, 8192, 1, 8), ("decode, 16 users, 8k", 16, 8192, 512, 64)]


def run(gpu: str, configs: list, gsm8k: bool) -> dict:
    import json, os, re, subprocess, urllib.request
    from concurrent.futures import ThreadPoolExecutor

    started = time.time()
    env = {**os.environ, "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"}
    rows, notes = [], []

    def note(s):
        notes.append(f"[{gpu}] {s}"); print(f"[{gpu}] {s}", flush=True)

    def start(label, model, extra):
        t0 = time.time()
        log_path = f"/tmp/serve-{label}.log"
        cmd = ["vllm", "serve", model, "--port", "8000", "--max-model-len", "16384", "--no-enable-prefix-caching", *extra]
        srv = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        while time.time() - t0 < 1200 and srv.poll() is None:
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        log = open(log_path).read()
        if srv.poll() is not None:
            note(f"== {label} {model} {extra} FAILED to start:\n{log[-4000:]}")
            return None, {}
        keep = [l[-220:] for l in log.splitlines()
                if re.search(r"Model loading took|GPU KV cache size|[Qq]uantiz|[Mm]arlin|fallback|not support|WARNING", l)]
        note(f"== {label} {model} {extra}: healthy after {time.time() - t0:.0f}s\n" + "\n".join(keep[:20]))
        w = re.search(r"Model loading took ([\d.]+) GiB", log)
        kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
        return srv, {"weight_gib": float(w.group(1)) if w else None,
                     "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None}

    def bench(tag, model, users, inp, out, prompts):
        cmd = ["vllm", "bench", "serve", "--model", model, "--port", "8000", "--dataset-name", "random",
               "--random-input-len", str(inp), "--random-output-len", str(out), "--num-prompts", str(prompts),
               "--max-concurrency", str(users), "--ignore-eos", "--seed", str(SEED),
               "--percentile-metrics", "ttft,tpot", "--metric-percentiles", "50",
               "--save-result", "--result-dir", "/tmp", "--result-filename", f"{tag}.json"]
        if subprocess.run(cmd, env=env, stdout=open(f"/tmp/{tag}.log", "w"), stderr=subprocess.STDOUT).returncode:
            note(f"bench {tag} FAILED:\n" + open(f"/tmp/{tag}.log").read()[-3000:])
            return None
        return json.load(open(f"/tmp/{tag}.json"))

    def ask(model, q):
        body = {"model": model, "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": q["question"] + "\nSolve step by step. End with: The answer is <number>."}]}
        req = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("answer is")[-1])
        got = nums[0].replace(",", "").rstrip(".") if nums else None
        want = q["answer"].split("####")[-1].strip().replace(",", "")
        return got is not None and float(got) == float(want)

    if gsm8k:
        from datasets import load_dataset
        questions = list(load_dataset("openai/gsm8k", "main", split="test"))

    for label, model, extra in configs:
        srv, info = start(label, model, extra)
        if srv is None:
            rows.append({"gpu": gpu, "config": label, "model": model, "case": "startup failed"})
            continue
        base = {"gpu": gpu, "config": label, "model": model, **info}
        bench(f"{label}-warmup", model, 16, 1024, 128, 32)  # compile and cudagraph warmup, discarded
        for run_no in range(1, REPEATS + 1):
            for case, users, inp, out, prompts in SPEED:
                r = bench(f"{label}-{users}-{inp}-{out}-r{run_no}", model, users, inp, out, prompts)
                if r:
                    row = {**base, "case": case, "run": run_no, "users": users, "input_len": inp, "output_len": out,
                           "completed": r["completed"], "mean_ttft_ms": round(r["mean_ttft_ms"], 1),
                           "mean_tpot_ms": round(r["mean_tpot_ms"], 3) if out > 1 else None,
                           "output_tokens_per_s": round(r["output_throughput"], 1)}
                    rows.append(row)
                    print("RESULT " + json.dumps(row), flush=True)
        if gsm8k:
            t0 = time.time()
            with ThreadPoolExecutor(64) as pool:
                ok = list(pool.map(lambda q: ask(model, q), questions))
            rows.append({**base, "case": "gsm8k", "completed": len(ok), "gsm8k_correct": sum(ok),
                         "gsm8k_pct": round(100 * sum(ok) / len(ok), 1)})
            note(f"{label} gsm8k: {sum(ok)}/{len(ok)} in {time.time() - t0:.0f}s")
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except Exception:
            srv.kill()
        time.sleep(5)

    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    note("versions: " + ", ".join(l for l in freeze.splitlines() if l.split("==")[0].lower() in ("vllm", "torch", "transformers", "flashinfer-python")))
    note("gpu: " + subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    note(f"container wall time: {time.time() - started:.0f}s")
    return {"rows": rows, "notes": notes}


@app.function(gpu="B200", cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def on_b200() -> dict:
    return run("B200", WEIGHTS + [KV_FP8], gsm8k=True)


@app.function(gpu="H100", cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def on_h100() -> dict:
    return run("H100", WEIGHTS, gsm8k=False)


@app.local_entrypoint()
def main():
    import csv
    calls = [on_b200.spawn(), on_h100.spawn()]
    results = [c.get() for c in calls]
    rows = [r for res in results for r in res["rows"]]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "results_formats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    notes = [n for res in results for n in res["notes"]]
    open(OUT / "results_formats_notes.txt", "w").write("\n".join(notes))
    print("\n".join(notes))
