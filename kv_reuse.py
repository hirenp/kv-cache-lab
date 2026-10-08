"""Reusing a KV cache: prefix-cache hits, eviction, and offloading to CPU memory and disk (SPEC.md V15).

    modal run kv_reuse.py
    modal run kv_reuse.py --only 'lmcache disk'   # one server, written to results_reuse_lmcache_disk*.

One H100, Qwen3-8B in BF16, vLLM with prefix caching on. Prompts are sent as token IDs, so every
length is exact: a document (random tokens, its own seed) followed by a 32-token question.
Time to first token comes from requests with max_tokens=1, each the only request running.
  hit vs miss:  for 1k, 4k and 16k documents, the first request for a document (miss), then the
                same document with a new question (hit), 3 documents per length
  eviction:     read a 16k document, then enough other 16k documents to fill the GPU's KV cache
                1.5 times over, then the first document again; twice
Eviction runs on three servers: GPU only, LMCache with CPU memory, and LMCache with a local disk
and a 20 GB CPU buffer, too small to hold the documents read after the first, so a hit after
eviction has to come from disk. (With a 2 GB buffer, LMCache stalled waiting for CPU space.) A request that takes over
2 minutes is recorded as a failure for that server, and the run goes on to the next one.
Writes results_reuse.csv and results_reuse_notes.txt from the local client; run without --detach.
"""
import pathlib
import time

import modal

VLLM_VERSION = "0.30.0"
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "lmcache==0.5.5")
)
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
app = modal.App("kv-reuse")
MODEL = "Qwen/Qwen3-8B"
OUT = pathlib.Path(__file__).parent

QUESTION = 32
LENGTHS = [1024, 4096, 16384]
DOCS_PER_LENGTH = 3
EVICT_DOC = 16384
LMCACHE = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
# (name, extra `vllm serve` args, extra environment)
SERVERS = [
    ("gpu only", [], {}),
    ("lmcache cpu", ["--kv-transfer-config", LMCACHE],
     {"LMCACHE_CHUNK_SIZE": "256", "LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "100"}),
    ("lmcache disk", ["--kv-transfer-config", LMCACHE],
     {"LMCACHE_CHUNK_SIZE": "256", "LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "20",
      "LMCACHE_LOCAL_DISK": "file:///tmp/lmcache-disk/", "LMCACHE_MAX_LOCAL_DISK_SIZE": "120"}),
]


@app.function(gpu="H100!", cpu=8.0, memory=196608, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def run(only: str = "") -> dict:
    import json, os, random, re, subprocess, urllib.request

    started = time.time()
    rows, notes = [], []

    def note(s):
        notes.append(s); print(s, flush=True)

    def doc(seed, n):
        rng = random.Random(seed)
        return [rng.randrange(1000, 100000) for _ in range(n)]

    def ttft(tokens):
        body = {"model": MODEL, "prompt": tokens, "max_tokens": 1, "temperature": 0}
        req = urllib.request.Request("http://localhost:8000/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        urllib.request.urlopen(req, timeout=120).read()
        return round((time.perf_counter() - t0) * 1000, 1)

    def measure(name, capacity):
        ttft(doc(999, 512))  # warm up compile and CUDA graphs, discarded
        if name == "gpu only":
            for n in LENGTHS:
                for d in range(DOCS_PER_LENGTH):
                    base = doc(n * 100 + d, n)
                    miss = ttft(base + doc(1, QUESTION))
                    hit = ttft(base + doc(2, QUESTION))
                    rows.append({"server": name, "case": "hit vs miss", "doc_tokens": n, "doc": d, "miss_ms": miss, "hit_ms": hit})
                    print("RESULT " + json.dumps(rows[-1]), flush=True)
        # Eviction: one document, then enough others to fill the GPU cache 1.5 times over, then back.
        others = int(1.5 * capacity / (EVICT_DOC + QUESTION)) + 1
        for rep in range(1, 3):
            first = doc(7000 + rep, EVICT_DOC)
            cold = ttft(first + doc(10, QUESTION))
            warm = ttft(first + doc(11, QUESTION))
            fill = [ttft(doc(8000 + rep * 1000 + i, EVICT_DOC) + doc(12, QUESTION)) for i in range(others)]
            back = ttft(first + doc(13, QUESTION))
            rows.append({"server": name, "case": "eviction", "doc_tokens": EVICT_DOC, "doc": rep, "miss_ms": cold, "hit_ms": warm,
                         "other_docs": others, "median_other_ms": sorted(fill)[len(fill) // 2], "after_eviction_ms": back})
            print("RESULT " + json.dumps(rows[-1]), flush=True)

    for name, extra, extra_env in [s for s in SERVERS if not only or s[0] == only]:
        env = {**os.environ, "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/tmp/vllm-cache", **extra_env}
        log_path = f"/tmp/serve-{name.replace(' ', '-')}.log"
        cmd = ["vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "20480", *extra]
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
            note(f"== {name}: server FAILED to start:\n{log[-4000:]}")
            continue
        kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
        capacity = int(kv.group(1).replace(",", "")) if kv else 400000
        note(f"== {name}: healthy after {time.time() - t0:.0f}s, GPU KV cache {capacity} tokens; env {extra_env}")
        try:
            measure(name, capacity)
        except Exception as e:  # a hung or failed request: record it and go on to the next server
            note(f"{name}: measurement FAILED: {type(e).__name__}: {str(e)[:300]}")
        log = open(log_path).read()
        keep = [l[-220:] for l in log.splitlines() if re.search(r"LMCache|lmcache", l) and not re.search(r"[Pp]in count", l)]
        note(f"{name}: {len(re.findall(r'[Pp]in count', log))} LMCache pin-count warnings; other LMCache lines ({len(keep)}): " + " | ".join(keep[-10:]))
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except Exception:
            srv.kill()
        time.sleep(5)

    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    note("versions: " + ", ".join(l for l in freeze.splitlines() if l.split("==")[0].lower() in ("vllm", "torch", "lmcache")))
    note("gpu: " + subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    note("cpus and memory: " + subprocess.run(["sh", "-c", "nproc; free -g | head -2"], capture_output=True, text=True).stdout.replace("\n", " "))
    note(f"container wall time: {time.time() - started:.0f}s")
    return {"rows": rows, "notes": notes}


@app.local_entrypoint()
def main(only: str = ""):
    import csv
    res = run.remote(only)
    if res["rows"]:
        fields = list(dict.fromkeys(k for r in res["rows"] for k in r))
        with open(OUT / ("results_reuse.csv" if not only else f"results_reuse_{only.replace(' ', '_')}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(res["rows"])
    open(OUT / ("results_reuse_notes.txt" if not only else f"results_reuse_{only.replace(' ', '_')}_notes.txt"), "w").write("\n".join(res["notes"]))
    print("\n".join(res["notes"]))
