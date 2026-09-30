"""What happens to requests in flight when an inference worker dies, and what does recovering them cost?

    modal run --detach kv_failover.py
    modal run kv_failover.py --collect

Two vLLM replicas on one two-GPU Modal machine; this script is the client and the router.
Replica A is killed (or frozen) partway through a streamed answer and replica B takes over,
either restarting the answer or continuing it (see SPEC.md V8). The run saves its results to
a Modal volume as it goes; --collect writes results_failover_*.csv and failover_logs/.
"""

import csv
import json
import pathlib
import statistics
import time

import modal

VLLM_VERSION = "0.30.0"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# vLLM compiles some GPU kernels at startup and needs nvcc, so start from NVIDIA's CUDA devel image.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
)
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
results = modal.Volume.from_name("kv-failover-results", create_if_missing=True)
app = modal.App("kv-failover")

OUTPUT_TOKENS = 512
FAIL_AFTER = 128


class Replica:
    """One `vllm serve` process on one GPU, in its own process group so it can be killed or frozen whole."""

    def __init__(self, name, gpu, port):
        self.name, self.gpu, self.port, self.proc, self.starts = name, gpu, port, None, 0

    def start(self):
        import os
        import subprocess
        self.starts += 1
        log = open(f"/tmp/{self.name}-{self.starts}.log", "w")
        # Prefix caching is off, so a replica never has a prompt cached from an earlier attempt.
        self.proc = subprocess.Popen(
            ["vllm", "serve", MODEL, "--port", str(self.port), "--max-model-len", "32768", "--no-enable-prefix-caching"],
            env={**os.environ, "HF_HOME": "/hf", "CUDA_VISIBLE_DEVICES": str(self.gpu)},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_healthy(self):
        import urllib.request
        t0 = time.time()
        while time.time() - t0 < 1800:
            try:
                urllib.request.urlopen(f"http://localhost:{self.port}/health", timeout=2)
                return round(time.time() - t0, 1)
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"{self.name} never became healthy")

    def signal(self, sig):
        import os
        os.killpg(self.proc.pid, sig)

    def restart(self):
        import signal
        try:
            self.signal(signal.SIGKILL)
        except ProcessLookupError:  # already killed by the experiment
            pass
        self.proc.wait()
        time.sleep(5)  # let the GPU memory go
        self.start()
        return self.wait_healthy()


def stream(port, prompt, max_tokens, on_token=None, timeout=120):
    """Stream a completion. Returns the text, the time of every chunk, and the error that ended it, if any."""
    import urllib.request
    # return_token_ids asks vLLM for the prompt's and each chunk's token IDs, so an answer can be continued exactly.
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stream": True,
            "return_token_ids": True}
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"start": time.time(), "text": "", "times": [], "error": None, "error_time": None, "prompt_ids": None, "ids": []}
    done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.decode().strip()
                if line == "data: [DONE]":
                    done = True
                if not line.startswith("data: ") or done:
                    continue
                choice = json.loads(line[6:])["choices"][0]
                out["text"] += choice["text"]
                out["ids"] += choice.get("token_ids") or []
                out["prompt_ids"] = out["prompt_ids"] or choice.get("prompt_token_ids")
                out["times"].append(time.time())
                if on_token:
                    on_token(len(out["times"]))
    except Exception as e:
        out["error"], out["error_time"] = type(e).__name__, time.time()
    # A server that dies mid-answer can close the connection cleanly: the stream just ends, with no
    # exception. Only the missing [DONE] marker shows that the answer was cut off.
    if not done and not out["error"]:
        out["error"], out["error_time"] = "StreamEndedWithoutDone", time.time()
    return out


def continuation(prompt, first):
    """The prompt for continuing `first` elsewhere: token IDs when vLLM returned them, else text."""
    if first["prompt_ids"] and len(first["ids"]) == len(first["times"]):
        return first["prompt_ids"] + first["ids"], "token_ids"
    return prompt + first["text"], "text"


def count_tokens(port, text):
    import urllib.request
    req = urllib.request.Request(f"http://localhost:{port}/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["count"]


def make_prompt(lines, seed=0):
    # Numbered lines, so the prompt is long but not one sentence repeated.
    return "".join(f"Note {seed}-{i}: the shipment for order {1000 + i} left warehouse {i % 7}.\n" for i in range(lines)) + "Summary:"


def save(name, data):
    pathlib.Path(f"/results/{name}.json").write_text(json.dumps(data))
    results.commit()


def single_request(a, b):
    """Experiment 1: one request on A, killed after FAIL_AFTER tokens, recovered on B both ways."""
    import signal
    rows = []
    for lines in (50, 400, 1260):  # about 1k, 8.7k and 28k tokens, at about 22 tokens a line
        prompt = make_prompt(lines)
        prompt_tokens = count_tokens(b.port, prompt)
        ref = stream(b.port, prompt, OUTPUT_TOKENS)
        killed = {}

        def kill_at(n):
            if n == FAIL_AFTER and not killed:
                killed["t"] = time.time()
                a.signal(signal.SIGKILL)
        fail = stream(a.port, prompt, OUTPUT_TOKENS, on_token=kill_at)
        base = {"prompt_tokens": prompt_tokens, "tokens_before_failure": len(fail["times"]),
                "error": fail["error"], "detect_s": round(fail["error_time"] - killed["t"], 3),
                "reference_ttft_s": round(ref["times"][0] - ref["start"], 3), "reference_total_s": round(ref["times"][-1] - ref["start"], 2)}
        restart = stream(b.port, prompt, OUTPUT_TOKENS)
        rows.append({**base, "strategy": "restart", "resume_ttft_s": round(restart["times"][0] - restart["start"], 3),
                     "finish_s": round(restart["times"][-1] - restart["start"], 2), "tokens_generated": len(restart["times"]),
                     "matches_reference": restart["text"] == ref["text"]})
        cont_prompt, cont_input = continuation(prompt, fail)
        cont = stream(b.port, cont_prompt, OUTPUT_TOKENS - len(fail["times"]))
        rows.append({**base, "strategy": "continue", "resume_ttft_s": round(cont["times"][0] - cont["start"], 3),
                     "finish_s": round(cont["times"][-1] - cont["start"], 2), "tokens_generated": len(cont["times"]),
                     "matches_reference": fail["text"] + cont["text"] == ref["text"], "continue_input": cont_input})
        rows[-1]["a_restart_s"] = a.restart()
    return rows


def hang(a):
    """SIGSTOP freezes A with its sockets open: the stream stops, and nothing errors until a timeout fires."""
    import signal
    import urllib.request
    frozen = {}

    def freeze_at(n):
        if n == FAIL_AFTER and not frozen:
            frozen["t"] = time.time()
            a.signal(signal.SIGSTOP)
    out = stream(a.port, make_prompt(50), OUTPUT_TOKENS, on_token=freeze_at, timeout=10)
    t = time.time()
    try:
        urllib.request.urlopen(f"http://localhost:{a.port}/health", timeout=5)
        health = "answered"
    except Exception as e:
        health = type(e).__name__
    row = {"tokens_before_freeze": len(out["times"]), "client_error": out["error"],
           "client_error_after_s": round(out["error_time"] - frozen["t"], 2), "client_read_timeout_s": 10,
           "health_check": health, "health_check_after_s": round(time.time() - t, 2), "health_timeout_s": 5}
    a.signal(signal.SIGCONT)
    row["a_restart_s"] = a.restart()
    return row


def under_load(a, b, strategy, users=8, prompt_lines=120, max_tokens=2048, kill_after_s=6.0):
    """Experiment 2: users on both replicas; A dies and its users move to B with one strategy."""
    import signal
    import threading
    killed = {}
    per_user = []

    def user(i, replica):
        prompt = make_prompt(prompt_lines, seed=100 * (replica.port % 10) + i)
        first = stream(replica.port, prompt, max_tokens)
        rec = {"strategy": strategy, "user": f"{replica.name}{i}", "role": "survivor" if replica is b else "moved",
               "first_times": first["times"], "error": first["error"]}
        if first["error"]:
            if strategy == "restart":
                second = stream(b.port, prompt, max_tokens)
            else:
                second = stream(b.port, continuation(prompt, first)[0], max_tokens - len(first["times"]))
            last = first["times"][-1] if first["times"] else first["start"]
            rec["gap_s"] = round(second["times"][0] - last, 3) if second["times"] else None
            rec["second_times"] = second["times"]
        per_user.append(rec)

    threads = [threading.Thread(target=user, args=(i, r)) for r in (a, b) for i in range(users)]
    for t in threads:
        t.start()
    time.sleep(kill_after_s)
    killed["t"] = time.time()
    a.signal(signal.SIGKILL)
    for t in threads:
        t.join()

    def gaps(times, lo, hi):
        return [1000 * (y - x) for x, y in zip(times, times[1:]) if lo <= x < hi]

    rows = []
    for rec in per_user:
        row = {k: rec[k] for k in ("strategy", "user", "role", "error")}
        row["gap_s"] = rec.get("gap_s")
        if rec["role"] == "survivor":
            before = gaps(rec["first_times"], 0, killed["t"])
            after = gaps(rec["first_times"], killed["t"], float("inf"))
            for tag, g in (("before", before), ("after", after)):
                if g:
                    row[f"itl_{tag}_p50_ms"] = round(statistics.median(g), 1)
                    row[f"itl_{tag}_p99_ms"] = round(sorted(g)[int(0.99 * (len(g) - 1))], 1)
                    row[f"itl_{tag}_max_ms"] = round(max(g), 1)
        rows.append(row)
    return rows


@app.function(gpu="H100!:2", image=image, volumes={"/hf": hf_cache, "/results": results}, timeout=3 * 3600,
              single_use_containers=True, cpu=16.0, memory=131072)
def run():
    a, b = Replica("a", 0, 8001), Replica("b", 1, 8002)
    a.start(); b.start()
    save("startup", {"a_health_s": a.wait_healthy(), "b_health_s": b.wait_healthy()})
    save("single", single_request(a, b))
    save("hang", hang(a))
    load_rows = []
    for strategy in ("restart", "continue"):
        load_rows += under_load(a, b, strategy)
        save("load", load_rows)
        a.restart()
    logs = {p.name: p.read_text() for p in pathlib.Path("/tmp").glob("[ab]-*.log")}
    save("logs", logs)


@app.local_entrypoint()
def main(collect: bool = False):
    if not collect:
        # With --detach, the run carries on after this client exits.
        print("started", run.spawn().object_id)
        return
    saved = {e.path.removesuffix(".json"): json.loads(b"".join(results.read_file(e.path))) for e in results.listdir("/")}
    logdir = pathlib.Path("failover_logs")
    logdir.mkdir(exist_ok=True)
    for name, text in saved.get("logs", {}).items():
        (logdir / name).write_text(text)
    for name, rows in (("single", saved.get("single", [])), ("load", saved.get("load", []))):
        if rows:
            fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("strategy", "user", "role"), k))
            with open(f"results_failover_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    print(json.dumps({k: v for k, v in saved.items() if k in ("startup", "hang")}, indent=2))
    for r in saved.get("single", []):
        print(json.dumps(r))
