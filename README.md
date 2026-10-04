# KV Cache Lab

Prefill vs decode on Apple Silicon, made visible. `kv_lab.py` runs SmolLM2-135M-Instruct one model call at a time, prints the Q/K/V projection shapes for each call, and shows the KV cache growing by one token per decode step.

Write-ups: [part 1](https://hiren.me/posts/watching-a-kv-cache-grow/) (`kv_lab.py`), [part 2](https://hiren.me/posts/watching-a-kv-cache-grow-part-2/) (`kv_bandwidth.py`), [part 3](https://hiren.me/posts/watching-a-kv-cache-grow-part-3/) (`kv_offload.py`). The [inference disaggregation](https://hiren.me/posts/inference-disaggregation-part-1/) series continues with `kv_interference.py` (part 1) and `kv_disagg.py` (part 2), and the [inference in production](https://hiren.me/posts/inference-in-production-part-1/) series with `kv_coldstart.py` (part 2), `kv_failover.py` (parts 3 and 4) and `kv_workload.py` (part 5). The spec the code was built from is in [SPEC.md](SPEC.md).

| script | runs on | notes |
|---|---|---|
| `kv_lab.py` | Mac (MPS) or CPU | seconds |
| `kv_bandwidth.py` | Mac (MPS), CPU, or one CUDA GPU | a few minutes |
| `kv_offload.py` | one CUDA GPU (Modal H100 via `modal_run.py`) | |
| `kv_interference.py`, `kv_disagg.py` | two CUDA GPUs on one machine (Modal `H100:2` via `modal_run.py`) | `kv_interference.py` took about 20 minutes |
| `kv_coldstart.py` | Modal, one and two H100s | about 30 H100-minutes per round of starts |
| `kv_failover.py` | Modal, two H100s (`--drain`: one) | about 30 minutes on two H100s; `--drain` about 10 on one |
| `kv_workload.py` | Modal, one H100 | about 25 minutes with `--interference-only`, 35 to 40 for everything |

The GPU times are rough wall-clock times from my runs, not billing. Check the usage page on Modal's dashboard after a run.

## Run

Tested with Python 3.11. Newer Python versions may work but are untested with the pinned torch version. If you don't have Python 3.11, `uv venv --python 3.11 .venv` fetches one and creates the venv.

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hardware_info.py
python kv_lab.py
```

Other options:

```
python kv_lab.py --prompt "Explain why the sky is blue"
python kv_lab.py --device cpu
python kv_lab.py --decode-steps 20
python kv_lab.py --dtype float32
```

The first run downloads the model (about 270 MB) from Hugging Face. Tested with Python 3.11, torch 2.14.0, and transformers 5.17.0 on an M4 Pro with macOS 26.6. The cache-inspection code uses the transformers 5.x cache API (`cache.layers[i].keys`), so keep the pinned versions.

## Running on Modal

Everything except `kv_lab.py` and `kv_bandwidth.py` needs CUDA GPUs; I ran those scripts on [Modal](https://modal.com).

```
pip install -r requirements.txt   # includes the modal client
modal setup                       # opens a browser to log in and stores a token
```

- Run every `modal run` command from the repo root; the scripts upload files by relative path.
- No Hugging Face token is needed: SmolLM2 and Qwen2.5 are ungated. Without one you'll see a rate-limit warning, which is harmless at this scale.
- The runs create Modal volumes that persist between runs: `kv-cache-lab-hf` (model weights; about 65 GB once the 32B model is downloaded), `kv-cache-lab-scratch`, `kv-coldstart-results` and `kv-failover-results`. Delete them when you're done with `modal volume delete <name>`.
- `kv_coldstart.py` and `kv_failover.py` run with `--detach`, so they keep going if your laptop sleeps. Collect the results afterwards with `--collect`. If one hangs, find it with `modal app list` and stop it with `modal app stop <app id>`; every function also has a timeout.

## What the output shows

**Prefill** is the first model call. It runs every prompt token through the model in a single forward pass. Each layer computes Q, K and V for all N prompt tokens at once, and the K and V tensors are stored in the KV cache. The logits for the last prompt token pick the first generated token.

**Decode** is every call after that. Each call feeds exactly one token, the one generated last, together with the cache from the previous call. The model computes Q, K and V for that one token only. The Q/K/V hook lines show a sequence dimension of N during prefill and 1 during decode.

**The KV cache** holds the keys and values of every token processed so far, per layer, with shape `[batch, kv_heads, sequence_length, head_dim]`. Attention for a new token needs the keys and values of all earlier tokens. The cache lets decode reuse them instead of recomputing them from the whole sequence on every step.

**The cache grows by one token per decode step** because each decode call processes one new token, and that token's K and V are appended to every layer. Nothing else changes: old entries are never rewritten. With 10 decode steps on a 5-token prompt, the cache goes 0 → 5 → 6 → ... → 15.

**Decode still reads the whole cache.** Computing new K/V is O(1) per step, but the new token's query attends to every cached key, so each step reads the entire cache. That per-step cost grows with context length. V2 measures it.

SmolLM2-135M uses grouped-query attention with 9 query heads and 3 KV heads. That is why `k_proj` and `v_proj` outputs are 192 wide while `q_proj` is 576, and why the cache stores 3 heads per layer. In bf16 each cached token costs 30 layers × 2 (K, V) × 3 heads × 64 dims × 2 bytes = 23,040 bytes. The lab measures this from the real tensors and checks it against the formula.

N decode steps produce N + 1 generated tokens: one from the prefill logits and one from each decode call. The last one is never fed back, so the final cache length is prompt + N.

## Reading the timings

At 135M parameters and a handful of tokens, Python and kernel-launch overhead dominate. Prefill and decode latencies land close together (around 8 to 10 ms each on an M4 Pro), so "prefill tokens/sec" says little about the hardware. In V1 the timings exist to validate the measurement harness: MPS is synchronized before and after each call, and hook output is buffered so terminal I/O stays out of the measured interval. V2 makes the numbers meaningful by growing the context.

The `PREFILL_START` / `DECODE_NN_START` markers are `time.perf_counter()` values, meant for lining up model calls with an Instruments / Metal System Trace capture later.

## Recompute check

After decoding, the lab recomputes every cached position in one uncached pass and compares the result with the cache built one token at a time. In exact arithmetic they are equal, which is why caching works. In floating point they are not bit-identical, most likely because the kernels for a 15-token call and a 1-token call add numbers up in a different order. On the M4 Pro in bf16, about half the elements differ (max 0.125 against values up to about 19), and two of the eleven greedy tokens flip where the top two logits were one bf16 step (0.125) apart. In fp32 the differences shrink to about 1e-5 and every token matches. Results repeat exactly across runs.

## V2: decode cost vs context length

```
python kv_bandwidth.py
python kv_bandwidth.py --lengths 512,8192,32768 --steps 20 --cache dynamic
```

`kv_bandwidth.py` measures the GPU's copy bandwidth, then for each context length (512 to 32,768 tokens) fills the cache with one prefill, times 50 decode steps, and fits a straight line of latency against KV bytes. It runs once with Transformers' default `DynamicCache` and once with a preallocated `StaticCache`, and writes every row to `results.csv`. The full run takes a few minutes. Lengths above 8,192 exceed the model's trained context, so the text is meaningless there, but the bytes each step moves are real.

Results on the M4 Pro (in `results.csv`):

| context | KV MiB | DynamicCache | StaticCache |
|---:|---:|---:|---:|
| 517 | 11 | 8.22 ms | 9.44 ms |
| 2,053 | 45 | 8.67 ms | 10.32 ms |
| 8,197 | 180 | 10.62 ms | 14.10 ms |
| 32,773 | 720 | 17.96 ms | 32.08 ms |

Measured copy bandwidth was 212 GB/s (Apple quotes 273 GB/s). The line fit gives an empty-cache cost of about 8 ms for both caches. The slope works out to 2.7 passes over the cache per decode step for `DynamicCache` and 6.4 for `StaticCache` (R² 0.997 and 0.998).

The passes come from how Transformers 5.17 handles each cache:

- `DynamicCache` appends with `torch.cat`, which reads the whole cache and writes a new copy every step, and then attention reads it. That's about 3 passes.
- `StaticCache` makes Transformers pass an attention mask to SDPA, which turns off grouped-query attention there, so `repeat_kv` expands K and V from 3 heads to 9 before attention on every step (read 3, write 9, then attention reads 9). That's about 7 passes. The `mask` and `KV expanded` columns in the output show this, detected by wrapping the attention function for one extra step after the timed ones. `attn keys` is measured at that step, so it includes the 50 timed tokens.

A thermal check repeats the first measurement at the end; drift was +2.2%.

`--cache all` adds two ablations that remove those copies: `dynamic-inplace` (a preallocated buffer written in place) and `static-patched` (attention over the filled slots only, no head expansion). Both give bit-identical output and drop to about 1.3 passes (`results_ablation.csv`).

## V3: moving a cache vs rebuilding it

```
modal run modal_run.py --script kv_offload.py --args "--net-dir /net" --out results_offload.csv
```

`kv_offload.py` needs a CUDA GPU; `modal_run.py` runs it on a Modal H100 (see "Running on Modal" above). For each model and context length it times a prefill (recompute), then copies the cache to GPU memory, pinned and pageable host RAM, local disk and a Modal Volume and back, and checks each reloaded cache decodes bit-identically. Results are in `results_offload.csv`: for Qwen2.5-7B at 32k tokens, recompute takes 1,201 ms and reload takes 68 ms (pinned RAM) to 563 ms (Volume). On H100, disable cuDNN attention for decode with a growing cache (`torch.backends.cuda.enable_cudnn_sdp(False)`); with it on, each decode step paid about 48 ms extra because the key length changes every step.

## Inference disaggregation (two GPUs)

Both scripts need two CUDA GPUs on one machine; `modal_run.py` runs them on two Modal H100s.

```
modal run modal_run.py --script kv_interference.py --out results_interference.csv
modal run modal_run.py --script kv_disagg.py --out results_disagg.csv
```

`kv_interference.py` (part 1): user A decodes while request B arrives with a long prompt. It records every gap between A's tokens and B's time to first token when B's prefill runs on A's GPU (colocated), in chunks between A's steps (chunked), or on the other GPU in a thread or a separate process. At 32k tokens, colocated stalls A for 1,472 ms; the separate process leaves A's worst gap at 26 ms. The thread version, which also copies B's cache to A's GPU, still stalls A (1,161 ms): the copy is queued on the stream A's decode uses and waits for B's prefill. The same thread without the copy (`disaggregated-thread-nocopy`, added later, `results_interference_threadcheck.csv`) leaves A's worst gap at 50 ms.

`kv_disagg.py` (part 2): prefill on GPU 0, decode on GPU 1. It measures the GPU-to-GPU link (395 GB/s over NVLink) and the cost of copying the cache after prefill or layer by layer during it. At 32k tokens the copy takes 5.7 ms, 0.5% of prefill; layer-by-layer copying didn't help.

## Inference in production: vLLM cold starts and failover

These run vLLM 0.30 on Modal H100s directly with `modal run`, not through `modal_run.py`.

```
# Part 2: where a new replica's startup time goes (7B on one H100, 32B on two)
modal run --detach kv_coldstart.py                     # one round, downloads the 32B weights first
modal run --detach kv_coldstart.py --repeats 3 --first-rep 1 --no-download-weights
modal run kv_coldstart.py --collect --out results_coldstart_repeats.csv

# Part 3: kill or freeze one of two replicas mid-answer and recover on the other
modal run --detach kv_failover.py
# Part 4: send SIGTERM to a replica with answers streaming, default vs --shutdown-timeout
modal run --detach kv_failover.py --drain
modal run kv_failover.py --collect

# Part 5: request shapes, and a decode stream with and without long prompts arriving,
# at four chunked-prefill settings. No --detach: see below.
modal run kv_workload.py --interference-only
```

Keep `--detach` on the runs: the command only starts the work and returns, and without `--detach` Modal stops the app as soon as it returns. The work saves results to a volume as each step finishes, and `--collect` is a second command that downloads them into CSVs and `coldstart_logs/` or `failover_logs/`.

`kv_workload.py` is the exception. It writes `results_workload*.csv` from the local client when the run ends, so run it without `--detach` and keep the terminal open; if the client exits, Modal stops the run.

## Notes

The prompt goes in as raw text. The chat template is deliberately skipped, because it would add about 30 system-prompt tokens around a 5-token prompt. Greedy decoding does not stop at the end-of-turn token, so longer runs can continue past `<|im_end|>` into a new chat turn. That output is expected.
