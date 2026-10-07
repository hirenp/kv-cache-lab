"""Speculative decoding: how many guessed tokens get accepted, and the speedup at 1, 8 and 64 users (SPEC.md V16).

    modal run kv_specdec.py

One H100, Qwen3-8B in BF16. Four servers, one at a time: no speculation, n-gram lookup in the
prompt on the CPU and on the GPU (4 tokens a step), and Qwen3-0.6B as a draft model (4 tokens a
step). Each server's startup lines about CUDA graphs, scheduling and speculation go in the notes. Two tasks with real
text, greedy (temperature 0), thinking off, up to 256 output tokens:
  gsm8k:   GSM8K test questions, so the answer is new text
  repeat:  a WikiText passage the model is asked to repeat and extend by one sentence, so the
           answer copies the prompt
For each task and each number of users at once (1, 8, 64): output tokens per second for one user
and for the server, and the acceptance counters from vLLM's /metrics. Outputs are compared with the
no-speculation server's. Writes results_specdec.csv and results_specdec_notes.txt from the local
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
app = modal.App("kv-specdec")
MODEL = "Qwen/Qwen3-8B"
OUT = pathlib.Path(__file__).parent

USERS = [1, 8, 64]
PROMPTS = {1: 16, 8: 48, 64: 192}
MAX_TOKENS = 256
SERVERS = [
    ("none", []),
    ("ngram", ["--speculative-config", '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}']),
    ("ngram on gpu", ["--speculative-config", '{"method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}']),
    ("draft model", ["--speculative-config", '{"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 4}']),
]


@app.function(gpu="H100!", cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def run() -> dict:
    import json, os, re, subprocess, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from datasets import load_dataset

    started = time.time()
    env = {**os.environ, "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"}
    rows, notes, answers = [], [], {}

    def note(s):
        notes.append(s); print(s, flush=True)

    gsm8k = [q["question"] + "\nSolve step by step." for q in load_dataset("openai/gsm8k", "main", split="test")]
    wiki = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if 600 < len(t) < 1200]
    tasks = {
        "gsm8k": gsm8k,
        "repeat": ["Repeat the following text exactly, then add one sentence that continues it.\n\n" + t.strip() for t in wiki],
    }

    def metrics():
        text = urllib.request.urlopen("http://localhost:8000/metrics", timeout=10).read().decode()
        out = {}
        for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
            vals = re.findall(rf"^vllm:spec_decode_{key}(?:_total)?(?:\{{[^}}]*\}})? ([\d.e+]+)$", text, re.M)
            out[key] = sum(float(v) for v in vals)
        return out

    def ask(prompt):
        body = {"model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=900))
        return time.perf_counter() - t0, r["usage"]["completion_tokens"], r["choices"][0]["message"]["content"]

    for name, extra in SERVERS:
        log_path = f"/tmp/serve-{name.replace(' ', '-')}.log"
        cmd = ["vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "4096", "--no-enable-prefix-caching", *extra]
        srv = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1200 and srv.poll() is None:
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        if srv.poll() is not None:
            note(f"== {name}: server FAILED to start:\n{open(log_path).read()[-4000:]}")
            continue
        note(f"== {name}: healthy after {time.time() - t0:.0f}s, args {extra}")
        startup = [l[-240:] for l in open(log_path).read().splitlines()
                   if re.search(r"[Cc]uda ?[Gg]raph|cudagraph|[Aa]sync|[Ss]peculat|[Ee]ager|[Ff]all(ing)? ?back|not supported|WARNING", l)]
        note(f"{name} startup lines ({len(startup)}):\n  " + "\n  ".join(startup[:40]))
        ask(tasks["gsm8k"][-1])  # warm up compile and CUDA graphs, discarded

        for task, prompts in tasks.items():
            for users in USERS:
                batch = prompts[:PROMPTS[users]]
                before = metrics()
                t0 = time.perf_counter()
                with ThreadPoolExecutor(users) as pool:
                    results = list(pool.map(ask, batch))
                wall = time.perf_counter() - t0
                after = metrics()
                drafts = after["num_drafts"] - before["num_drafts"]
                drafted = after["num_draft_tokens"] - before["num_draft_tokens"]
                accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
                tokens = sum(r[1] for r in results)
                texts = [r[2] for r in results]
                key = (task, users)
                if name == "none":
                    answers[key] = texts
                same = sum(a == b for a, b in zip(texts, answers.get(key, []))) if key in answers else None
                rows.append({"server": name, "task": task, "users": users, "requests": len(batch), "output_tokens": tokens,
                             "per_user_tokens_per_s": round(sum(r[1] / r[0] for r in results) / len(results), 1),
                             "server_tokens_per_s": round(tokens / wall, 1),
                             "draft_tokens": int(drafted), "accepted_tokens": int(accepted),
                             "acceptance_rate": round(accepted / drafted, 3) if drafted else None,
                             "accepted_per_step": round(accepted / drafts, 2) if drafts else None,
                             "same_output_as_none": same})
                print("RESULT " + json.dumps(rows[-1]), flush=True)
        srv.terminate()
        try:
            srv.wait(timeout=120)
        except Exception:
            srv.kill()
        time.sleep(5)

    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    note("versions: " + ", ".join(l for l in freeze.splitlines() if l.split("==")[0].lower() in ("vllm", "torch", "transformers")))
    note("gpu: " + subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    note(f"container wall time: {time.time() - started:.0f}s")
    return {"rows": rows, "notes": notes}


@app.local_entrypoint()
def main():
    import csv
    res = run.remote()
    if res["rows"]:
        with open(OUT / "results_specdec.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0]))
            w.writeheader()
            w.writerows(res["rows"])
    open(OUT / "results_specdec_notes.txt", "w").write("\n".join(res["notes"]))
    print("\n".join(res["notes"]))
