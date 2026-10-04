"""The same model served in 16 bits and in FP4: memory, decode speed, accuracy (SPEC.md V11).

    modal run kv_fp4_serve.py

One GPU, one `vllm serve` at a time, `vllm bench serve` and a GSM8K check against it from the
same container. For each checkpoint:
  memory:   weight memory from vLLM's "Model loading took" log line
  speed:    mean time per output token at 1 user (256-token prompts) and at 16 users with
            256-token and 8,192-token prompts, so the KV cache is small and then large
  accuracy: the 1,319 GSM8K test questions, greedy, and how many answers match the 16-bit run
Each speed run twice. Writes results_fp4_serve.csv, results_fp4_answers.jsonl and
results_fp4_serve_notes.txt from the local client, so run it without --detach.
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
app = modal.App("kv-fp4-serve")
GPU = "B200"
CHECKPOINTS = [("16-bit", "Qwen/Qwen3-8B"), ("fp4", "RedHatAI/Qwen3-8B-NVFP4")]
OUT = pathlib.Path(__file__).parent

SEED = 0
REPEATS = 2
# (case, users, input tokens, output tokens, prompts)
SPEED = [("1 user", 1, 256, 512, 8), ("16 users, short", 16, 256, 512, 64), ("16 users, 8k", 16, 8192, 512, 64)]


@app.function(gpu=GPU, cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache}, timeout=3600)
def experiment() -> dict:
    import json, os, re, subprocess, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from datasets import load_dataset

    started = time.time()
    env = {**os.environ, "HF_HOME": "/hf", "VLLM_CACHE_ROOT": "/tmp/vllm-cache"}
    rows, answers, notes = [], {}, []
    questions = list(load_dataset("openai/gsm8k", "main", split="test"))

    def note(s):
        notes.append(s); print(s, flush=True)

    def start(label, model):
        t0 = time.time()
        log_path = f"/tmp/serve-{label}.log"
        cmd = ["vllm", "serve", model, "--port", "8000", "--max-model-len", "16384", "--no-enable-prefix-caching"]
        srv = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        while time.time() - t0 < 1200 and srv.poll() is None:
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        log = open(log_path).read()
        if srv.poll() is not None:
            note(f"== {label} {model} FAILED to start:\n{log[-4000:]}")
            return None, None
        keep = [l[-220:] for l in log.splitlines() if re.search(r"Model loading took|quantization|KV cache|Using .*(backend|kernel)", l)]
        note(f"== {label} {model}: healthy after {time.time() - t0:.0f}s\n" + "\n".join(keep[:15]))
        m = re.search(r"Model loading took ([\d.]+) GiB", log)
        return srv, float(m.group(1)) if m else None

    def bench(tag, model, users, inp, out, prompts):
        cmd = ["vllm", "bench", "serve", "--model", model, "--port", "8000", "--dataset-name", "random",
               "--random-input-len", str(inp), "--random-output-len", str(out), "--num-prompts", str(prompts),
               "--max-concurrency", str(users), "--ignore-eos", "--seed", str(SEED),
               "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "50,99",
               "--save-result", "--result-dir", "/tmp", "--result-filename", f"{tag}.json"]
        if subprocess.run(cmd, env=env, stdout=open(f"/tmp/{tag}.log", "w"), stderr=subprocess.STDOUT).returncode:
            note(f"bench {tag} FAILED:\n" + open(f"/tmp/{tag}.log").read()[-3000:])
            return None
        return json.load(open(f"/tmp/{tag}.json"))

    def ask(model, q):
        body = {"model": model, "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}, "messages": [
            {"role": "user", "content": q["question"] + "\nSolve step by step. End with: The answer is <number>."}]}
        req = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.split("answer is")[-1])
        got = nums[0].replace(",", "").rstrip(".") if nums else None
        want = q["answer"].split("####")[-1].strip().replace(",", "")
        return got, got is not None and float(got) == float(want)

    for label, model in CHECKPOINTS:
        srv, weight_gib = start(label, model)
        if srv is None:
            continue
        bench(f"{label}-warmup", model, 16, 1024, 128, 32)  # compile and cudagraph warmup, discarded
        for run in range(1, REPEATS + 1):
            for case, users, inp, out, prompts in SPEED:
                r = bench(f"{label}-{users}-{inp}-r{run}", model, users, inp, out, prompts)
                if r:
                    row = {"checkpoint": label, "model": model, "case": case, "run": run, "users": users,
                           "input_len": inp, "output_len": out, "completed": r["completed"],
                           "weight_gib": weight_gib, "mean_tpot_ms": round(r["mean_tpot_ms"], 3),
                           "tokens_per_s_per_user": round(1000 / r["mean_tpot_ms"], 1),
                           "output_tokens_per_s": round(r["output_throughput"], 1),
                           "mean_ttft_ms": round(r["mean_ttft_ms"], 1)}
                    rows.append(row)
                    print("RESULT " + json.dumps(row), flush=True)

        t0 = time.time()
        with ThreadPoolExecutor(64) as pool:
            res = list(pool.map(lambda q: ask(model, q), questions))
        answers[label] = [g for g, _ in res]
        correct = sum(ok for _, ok in res)
        row = {"checkpoint": label, "model": model, "case": "gsm8k", "completed": len(res),
               "weight_gib": weight_gib, "gsm8k_correct": correct, "gsm8k_pct": round(100 * correct / len(res), 1)}
        if label != CHECKPOINTS[0][0] and CHECKPOINTS[0][0] in answers:
            row["same_answer_as_16bit_pct"] = round(100 * sum(a == b for a, b in zip(answers[label], answers[CHECKPOINTS[0][0]])) / len(res), 1)
        rows.append(row)
        note(f"{label} gsm8k: {correct}/{len(res)} in {time.time() - t0:.0f}s")

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
    return {"rows": rows, "answers": answers, "notes": notes}


@app.local_entrypoint()
def main():
    import csv, json
    result = experiment.remote()
    rows = result["rows"]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "results_fp4_serve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "results_fp4_answers.jsonl", "w") as f:
        for label, a in result["answers"].items():
            f.write(json.dumps({"checkpoint": label, "answers": a}) + "\n")
    open(OUT / "results_fp4_serve_notes.txt", "w").write("\n".join(result["notes"]))
    print("\n".join(result["notes"]))
