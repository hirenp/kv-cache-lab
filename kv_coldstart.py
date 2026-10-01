"""How long does a new vLLM replica take to become useful, and where does the time go?

    modal run --detach kv_coldstart.py --repeats 3 --first-rep 1 --no-download-weights
    modal run kv_coldstart.py --collect --out results_coldstart_repeats.csv

Runs every start in a fresh Modal container (see SPEC.md V7). A Modal function runs the
starts one after another and saves each result to a volume as it finishes, so a long run
survives the local client disconnecting. --collect then writes the CSV and the full vLLM
logs to coldstart_logs/.
"""

import csv
import json
import pathlib
import re
import time

import modal

VLLM_VERSION = "0.30.0"
MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
MODEL_32B = "Qwen/Qwen2.5-32B-Instruct"

# vLLM compiles some GPU kernels at startup and needs nvcc, so start from NVIDIA's CUDA devel image.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
)
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
results = modal.Volume.from_name("kv-coldstart-results", create_if_missing=True)
app = modal.App("kv-coldstart")

# single_use_containers: every call gets a new container, so no compile or kernel cache carries over.
OPTIONS = dict(image=image, volumes={"/hf": hf_cache}, timeout=3600, single_use_containers=True, cpu=8.0, memory=65536)
CACHE_DIRS = ["/root", "/tmp", "/usr/local/lib/python3.12/site-packages"]


def serve_and_measure(name, model, args, prefix_test=False):
    """Start `vllm serve`, wait for /health, time a first token and decode, then stop it."""
    import os
    import subprocess
    import urllib.request

    def post(body):
        req = urllib.request.Request("http://localhost:8000/v1/completions",
                                     data=json.dumps({"model": model, "temperature": 0, **body}).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=900)

    def first_token(prompt):
        t = time.time()
        r = post({"prompt": prompt, "max_tokens": 16, "stream": True})
        r.readline()
        elapsed = time.time() - t
        r.read()
        return round(elapsed, 3)

    log_path = f"/tmp/{name}.log"
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.split("\n")[0]
    t0 = time.time()
    with open(log_path, "w") as log:  # the server keeps its own copy of the file handle
        proc = subprocess.Popen(["vllm", "serve", model, "--port", "8000", "--max-model-len", "32768", *args],
                                env={**os.environ, "HF_HOME": "/hf"}, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    row = {"run": name, "model": model.split("/")[1], "args": " ".join(args), "gpu": gpu, "start_epoch": round(t0, 1)}
    try:
        while time.time() - t0 < 1800 and proc.poll() is None:
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                row["health_s"] = round(time.time() - t0, 1)
                break
            except Exception:
                time.sleep(0.5)
        if "health_s" in row:
            row["first_token_s"] = first_token("The capital of France is")
            t = time.time()
            usage = json.loads(post({"prompt": "Write a long story about a lighthouse.", "max_tokens": 256, "ignore_eos": True}).read())["usage"]
            row["decode_tok_s"] = round(usage["completion_tokens"] / (time.time() - t), 1)
            if prefix_test:
                # The same ~30k-token prompt twice: the first computes it all, the second hits the prefix cache.
                prompt = "The quick brown fox jumps over the lazy dog. " * 3000
                row["long_prompt_first_s"] = first_token(prompt)
                row["long_prompt_repeat_s"] = first_token(prompt)
    except Exception as e:  # keep the log and the timings measured so far
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            proc.wait()
        time.sleep(5)
    return row, open(log_path).read()


def files_written_since(t0):
    """Group files written after t0 by directory: total size and first/last write, seconds after t0."""
    import os
    groups = {}
    for top in CACHE_DIRS:
        for dirpath, _, files in os.walk(top):
            for f in files:
                try:
                    st = os.stat(os.path.join(dirpath, f))
                except OSError:
                    continue
                if st.st_mtime < t0:
                    continue
                key = "/".join(dirpath.split("/")[:6])
                g = groups.setdefault(key, [0, 0, float("inf"), 0])
                g[0] += 1; g[1] += st.st_size
                g[2] = min(g[2], st.st_mtime - t0); g[3] = max(g[3], st.st_mtime - t0)
    lines = [f"{v[2]:7.1f} s .. {v[3]:7.1f} s  {v[0]:6d} files  {v[1] / 1e6:9.1f} MB  {k}" for k, v in groups.items()]
    return "\n".join(sorted(lines))


def cold_then_warm(model, args, warm, rep):
    rows, logs = [], {}
    tag = model.split("-")[1].lower() + ("-eager" if "--enforce-eager" in args else "")
    row, logs[f"{tag}-cold-r{rep}"] = serve_and_measure(f"{tag}-cold", model, args)
    logs[f"{tag}-cold-r{rep}_files"] = files_written_since(row["start_epoch"])
    rows.append(row)
    if warm:
        row, logs[f"{tag}-warm-r{rep}"] = serve_and_measure(f"{tag}-warm", model, args, prefix_test=model == MODEL_7B)
        rows.append(row)
    for row in rows:
        row["rep"] = rep
    return rows, logs


# "H100!" pins the GPU type: plain "H100" lets Modal hand out an H200, which decodes faster.
@app.function(gpu="H100!", **OPTIONS)
def one_gpu(model: str, args: list, warm: bool, rep: int):
    return cold_then_warm(model, args, warm, rep)


@app.function(gpu="H100!:2", **OPTIONS)
def two_gpus(model: str, args: list, warm: bool, rep: int):
    return cold_then_warm(model, args, warm, rep)


@app.function(**OPTIONS)
def download(model: str) -> dict:
    import os
    os.environ["HF_HOME"] = "/hf"
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    seconds = time.time() - t
    size = sum(p.stat().st_size for p in pathlib.Path(path).rglob("*") if p.is_file())
    hf_cache.commit()
    return {"run": "32b-download", "model": model.split("/")[1], "download_s": round(seconds, 1), "download_gb": round(size / 1e9, 1)}


# Phase timings vLLM prints at startup.
PATTERNS = {
    "weights_s": r"Loading weights took ([\d.]+) seconds",
    "model_load_s": r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds",
    "compile_s": r"torch\.compile took ([\d.]+) s in total",
    "engine_init_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
    "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
}


def parse(log):
    out = {}
    for key, pat in PATTERNS.items():
        m = re.findall(pat, log)
        if m:
            out[key] = float(m[0]) if key != "compile_s" else sum(map(float, m))
    graphs = re.findall(r"Graph capturing finished in ([\d.]+) secs", log)
    if graphs:
        out["graph_capture_s"] = sum(map(float, graphs))
    return out


# One start at a time, so parallel starts don't share the volume's read bandwidth.
@app.function(volumes={"/results": results}, timeout=6 * 3600)
def run_all(first_rep: int, repeats: int, download_weights: bool):
    jobs = [(one_gpu, MODEL_7B, [], True), (one_gpu, MODEL_7B, ["--enforce-eager"], False),
            (two_gpus, MODEL_32B, ["--tensor-parallel-size", "2"], True)]
    if download_weights:
        # The 32B weights have to be in the volume before its start is timed, so the download goes first.
        pathlib.Path("/results/32b-download.json").write_text(json.dumps({"rows": [download.remote(MODEL_32B)], "logs": {}}))
        results.commit()
    for rep in range(first_rep, first_rep + repeats):
        for fn, model, args, warm in jobs:
            try:
                rows, logs = fn.remote(model, args, warm, rep)
            except Exception as e:  # one failed start shouldn't lose the rest of the run
                print(f"rep {rep} {model} {args} failed: {e!r}")
                continue
            name = f"{rows[0]['run']}-r{rep}.json"
            pathlib.Path(f"/results/{name}").write_text(json.dumps({"rows": rows, "logs": logs}))
            results.commit()


@app.local_entrypoint()
def main(out: str = "results_coldstart.csv", repeats: int = 1, first_rep: int = 0, download_weights: bool = True,
         collect: bool = False):
    if not collect:
        # With --detach, the run carries on after this client exits.
        print("started", run_all.spawn(first_rep, repeats, download_weights).object_id)
        return
    logdir = pathlib.Path("coldstart_logs")
    logdir.mkdir(exist_ok=True)
    rows = []
    for entry in results.listdir("/"):
        saved = json.loads(b"".join(results.read_file(entry.path)))
        for name, text in saved["logs"].items():
            (logdir / (name + (".txt" if name.endswith("_files") else ".log"))).write_text(text)
        for row in saved["rows"]:
            if "rep" in row:
                row.update(parse(saved["logs"][f"{row['run']}-r{row['rep']}"]))
            rows.append(row)
    rows.sort(key=lambda r: (r.get("rep", -1), r["run"]))
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r))
