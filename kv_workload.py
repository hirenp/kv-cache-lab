"""How request shapes and long prefills change decode latency on one GPU (SPEC.md V10).

    modal run kv_workload.py                       # shapes and interference, all configs
    modal run kv_workload.py --interference-only   # what the part 5 interference table used
    modal run kv_workload.py --interference-only --configs no-chunk-16384,chunk-2048

One H100, one `vllm serve`, `vllm bench serve` against it from the same container.
For each server config (chunked prefill off, and on at several token budgets):
  shapes:       8192/256, 1024/1024, 256/4096 at fixed concurrency, each alone
  interference: a 256/1024 decode stream alone, then again while a second client
                sends 8192/16 requests at a steady rate
Each measurement runs twice. Unlike kv_coldstart.py and kv_failover.py, this script writes its
results from the local client, so run it without --detach and keep the terminal open until it
finishes: results_workload[_interference].csv and the matching _notes.txt.
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
app = modal.App("kv-workload-shapes")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = pathlib.Path(__file__).parent

CONCURRENCY = 16
SEED = 0
SHAPES = [(8192, 256, 64), (1024, 1024, 64), (256, 4096, 32)]  # input, output, prompts
DECODE = (256, 1024, 64)
PREFILL_RATE = 1.0  # interfering 8192/16 requests per second
# (name, extra `vllm serve` args, run the shapes too)
CONFIGS = [
    ("no-chunk-16384", ["--max-num-batched-tokens", "16384", "--no-enable-chunked-prefill"], True),
    ("chunk-2048", ["--max-num-batched-tokens", "2048"], True),
    ("chunk-8192-default", [], False),
    ("chunk-512", ["--max-num-batched-tokens", "512"], False),
]
REPEATS = 2


@app.function(gpu="H100", cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def experiment(interference_only: bool, configs: str) -> dict:
    import json, os, subprocess, urllib.request
    started = time.time()
    env = {**os.environ, "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"}
    rows, notes = [], []

    def note(s):
        notes.append(s); print(s, flush=True)

    def bench(tag, inp, out, prompts, extra):
        path = f"/tmp/{tag}.json"
        cmd = ["vllm", "bench", "serve", "--model", MODEL, "--port", "8000", "--dataset-name", "random",
               "--random-input-len", str(inp), "--random-output-len", str(out), "--num-prompts", str(prompts),
               "--ignore-eos", "--seed", str(SEED), "--percentile-metrics", "ttft,tpot,itl,e2el",
               "--metric-percentiles", "50,90,99", "--save-result", "--save-detailed",
               "--result-dir", "/tmp", "--result-filename", f"{tag}.json", *extra]
        return subprocess.Popen(cmd, env=env, stdout=open(f"/tmp/{tag}.log", "w"), stderr=subprocess.STDOUT), path

    def prompt_tokens():
        # Server-side count of prompt tokens prefilled so far, from vLLM's Prometheus counter.
        text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5).read().decode()
        return sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith("vllm:prompt_tokens_total"))

    def record(config, kind, run, case, path, inp, out, prefills=None):
        r = json.load(open(path))
        itls = sorted(x * 1000 for req in r.get("itls", []) for x in req) or [float("nan")]
        row = {"config": config, "kind": kind, "run": run, "case": case, "input_len": inp, "output_len": out,
               "concurrency": CONCURRENCY, "completed": r["completed"], "duration_s": round(r["duration"], 1),
               "prefills_8k_during_run": prefills,
               "itl_samples": len(itls), "p999_itl_ms": round(itls[int(0.999 * (len(itls) - 1))], 1),
               "max_itl_ms": round(itls[-1], 1)}
        for m in ("mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p50_tpot_ms", "p99_tpot_ms",
                  "mean_itl_ms", "p50_itl_ms", "p90_itl_ms", "p99_itl_ms", "mean_e2el_ms", "p99_e2el_ms"):
            row[m] = round(r[m], 2) if m in r else None
        rows.append(row)
        print("RESULT " + json.dumps(row), flush=True)

    def run_bench(tag, inp, out, prompts, extra=()):
        p, path = bench(tag, inp, out, prompts, list(extra) + ["--max-concurrency", str(CONCURRENCY)])
        if p.wait() != 0:
            note(f"bench {tag} FAILED:\n" + open(f"/tmp/{tag}.log").read()[-3000:])
            return None
        return path

    def start(name, args):
        t0 = time.time()
        log_path = f"/tmp/serve-{name}.log"
        serve = ["vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "16384", "--no-enable-prefix-caching", *args]
        srv = subprocess.Popen(serve, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        healthy = False
        while time.time() - t0 < 1200 and srv.poll() is None:
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                healthy = True
                break
            except Exception:
                time.sleep(2)
        sched = [l for l in open(log_path).read().splitlines() if "chunked prefill" in l.lower() or "enable_chunked_prefill=" in l]
        note(f"== {name} {args}: healthy={healthy} after {time.time() - t0:.0f}s; scheduler lines: {[l[-200:] for l in sched[:3]]}")
        if not healthy:
            note(open(log_path).read()[-4000:])
            srv.kill()
            return None
        return srv

    wanted = [c for c in CONFIGS if not configs or c[0] in configs.split(",")]
    for name, args, do_shapes in wanted:
        do_shapes = do_shapes and not interference_only
        srv = start(name, args)
        if srv is None and "--no-enable-chunked-prefill" in args:
            srv = start(name, [a for a in args if a != "--no-enable-chunked-prefill"])
        if srv is None:
            continue

        run_bench(f"{name}-warmup", 1024, 128, 32)  # compile and cudagraph warmup, discarded
        for run in range(1, REPEATS + 1):
            if do_shapes:
                for inp, out, prompts in SHAPES:
                    case = f"{inp}/{out}"
                    path = run_bench(f"{name}-shape-{inp}-{out}-r{run}", inp, out, prompts)
                    if path:
                        record(name, "shape", run, case, path, inp, out)
            inp, out, prompts = DECODE
            path = run_bench(f"{name}-decode-alone-r{run}", inp, out, prompts)
            if path:
                record(name, "interference", run, "decode alone", path, inp, out)
            # Steady 8192/16 arrivals: a gamma interval with burstiness 100 is close to constant.
            # The client spends a while generating prompts before it sends anything, so wait until
            # the server has prefilled the first 8k prompt before starting the measured stream.
            # 120 prompts is more than the stream needs; it's stopped when the stream ends.
            base = prompt_tokens()
            pre, _ = bench(f"{name}-prefill-r{run}", 8192, 16, 120, ["--request-rate", str(PREFILL_RATE), "--burstiness", "100"])
            t0 = time.time()
            while prompt_tokens() - base < 8192 and time.time() - t0 < 300 and pre.poll() is None:
                time.sleep(1)
            note(f"{name} r{run}: first 8k prefill after {time.time() - t0:.0f}s")
            before = prompt_tokens()
            path = run_bench(f"{name}-decode-with-prefills-r{run}", inp, out, prompts)
            # Prompt tokens prefilled during the stream, minus the stream's own 256-token prompts.
            prefills = round((prompt_tokens() - before - prompts * inp) / 8192, 1)
            pre.terminate()
            pre.wait(timeout=60)
            note(f"{name} r{run} prefill client log tail:\n" + open(f"/tmp/{name}-prefill-r{run}.log").read()[-1500:])
            if path:
                record(name, "interference", run, "decode + 8192/16 prefills", path, inp, out, prefills)
            time.sleep(5)
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


@app.local_entrypoint()
def main(interference_only: bool = False, configs: str = ""):
    import csv
    result = experiment.remote(interference_only, configs)
    rows = result["rows"]
    stem = "results_workload_interference" if interference_only else "results_workload"
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    open(OUT / f"{stem}_notes.txt", "w").write("\n".join(result["notes"]))
    print("\n".join(result["notes"]))
