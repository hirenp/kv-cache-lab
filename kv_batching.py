"""How one GPU serves more users at once: per-user speed against total throughput (SPEC.md V14).

    modal run kv_batching.py

One H100, Qwen3-8B in BF16, one `vllm serve`, `vllm bench serve` against it from the same container.
Every request has a 512-token prompt and a 256-token answer. For each number of users at once
(1 to 512), twice: mean and median time per output token, time to first token, output tokens per
second, and the server's preemption count (requests pushed out when the KV cache is full).
Writes results_batching.csv and results_batching_notes.txt from the local client, so run it
without --detach.
"""
import pathlib
import time

import modal

VLLM_VERSION = "0.30.0"
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
)
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
app = modal.App("kv-batching")
MODEL = "Qwen/Qwen3-8B"
OUT = pathlib.Path(__file__).parent

INPUT, OUTPUT = 512, 256
USERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SEED = 0
REPEATS = 2


@app.function(gpu="H100!", cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def run() -> dict:
    import json, os, re, subprocess, urllib.request

    started = time.time()
    env = {**os.environ, "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"}
    rows, notes = [], []

    def note(s):
        notes.append(s); print(s, flush=True)

    def preemptions():
        text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=10).read().decode()
        return sum(float(m.group(1)) for m in re.finditer(r"^vllm:num_preemptions(?:_total)?\{[^}]*\} ([\d.e+]+)$", text, re.M))

    log_path = "/tmp/serve.log"
    cmd = ["vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "4096", "--max-num-seqs", str(max(USERS)),
           "--no-enable-prefix-caching"]
    srv = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 1200 and srv.poll() is None:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            break
        except Exception:
            time.sleep(2)
    log = open(log_path).read()
    if srv.poll() is not None:
        note("server FAILED to start:\n" + log[-4000:])
        return {"rows": rows, "notes": notes}
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    note(f"server: {' '.join(cmd)}; healthy after {time.time() - t0:.0f}s")
    note("log: " + " | ".join(l[-200:] for l in log.splitlines() if re.search(r"Model loading took|GPU KV cache size|Maximum concurrency", l)))
    kv_tokens = int(kv.group(1).replace(",", "")) if kv else None

    def bench(tag, users, prompts):
        cmd = ["vllm", "bench", "serve", "--model", MODEL, "--port", "8000", "--dataset-name", "random",
               "--random-input-len", str(INPUT), "--random-output-len", str(OUTPUT), "--num-prompts", str(prompts),
               "--max-concurrency", str(users), "--ignore-eos", "--seed", str(SEED),
               "--percentile-metrics", "ttft,tpot", "--metric-percentiles", "50,99",
               "--save-result", "--result-dir", "/tmp", "--result-filename", f"{tag}.json"]
        if subprocess.run(cmd, env=env, stdout=open(f"/tmp/{tag}.log", "w"), stderr=subprocess.STDOUT).returncode:
            note(f"bench {tag} FAILED:\n" + open(f"/tmp/{tag}.log").read()[-3000:])
            return None
        return json.load(open(f"/tmp/{tag}.json"))

    bench("warmup", 16, 32)  # compile and CUDA graph warmup, discarded
    for run_no in range(1, REPEATS + 1):
        for users in USERS:
            prompts = max(32, 4 * users)
            before = preemptions()
            r = bench(f"u{users}-r{run_no}", users, prompts)
            if not r:
                continue
            row = {"run": run_no, "users": users, "prompts": prompts, "completed": r["completed"],
                   "mean_tpot_ms": round(r["mean_tpot_ms"], 2), "median_tpot_ms": round(r["median_tpot_ms"], 2),
                   "p99_tpot_ms": round(r["p99_tpot_ms"], 2), "mean_ttft_ms": round(r["mean_ttft_ms"], 1),
                   "output_tokens_per_s": round(r["output_throughput"], 1),
                   "per_user_tokens_per_s": round(1000 / r["mean_tpot_ms"], 1),
                   "preemptions": int(preemptions() - before), "kv_cache_tokens": kv_tokens}
            rows.append(row)
            print("RESULT " + json.dumps(row), flush=True)

    srv.terminate()
    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    note("versions: " + ", ".join(l for l in freeze.splitlines() if l.split("==")[0].lower() in ("vllm", "torch", "transformers", "flashinfer-python")))
    note("gpu: " + subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    note(f"container wall time: {time.time() - started:.0f}s")
    return {"rows": rows, "notes": notes}


@app.local_entrypoint()
def main():
    import csv
    res = run.remote()
    if res["rows"]:
        with open(OUT / "results_batching.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
            w.writeheader()
            w.writerows(res["rows"])
    open(OUT / "results_batching_notes.txt", "w").write("\n".join(res["notes"]))
    print("\n".join(res["notes"]))
