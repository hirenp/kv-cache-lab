# KV Cache Lab V1.1

Prefill vs Decode on Apple Silicon. Educational implementation specification for a coding agent.

**Goal:** Build a small Python program that makes transformer inference mechanics visible on an Apple Silicon Mac. It should clearly show the difference between prefill and decode, expose Q/K/V projection shapes, and show the KV cache growing one token at a time during autoregressive generation.

```
prompt
  |
PREFILL
  |
initial KV cache
  |
DECODE token 1 -> KV grows by 1
  |
DECODE token 2 -> KV grows by 1
  |
...
```

**Example values in this spec are illustrative.** Token counts, token IDs, shapes, and byte counts shown below were not produced by running the model. The program must print values read from the real model, tokenizer, and tensors. Do not try to match the examples.

### Changes from V1

- Pinned dtype (bf16) and exact dependency versions; dropped "support both cache formats".
- Raw prompt, no chat template.
- Defined the generated-token count (N decode calls produce N+1 generated tokens).
- Q/K/V logging now leads with output shapes so GQA is visible.
- Warmup covers one decode step as well as prefill.
- `head_dim` is read from config and cross-checked against the projection weights.
- `hardware_info.py` exposes a function that `kv_lab.py` calls with the selected device.
- The summary and README state that decode still reads the entire cache, and that timing at 135M is overhead-dominated.
- Examples use a placeholder prompt length N (the default prompt is probably 5 tokens, not 7).

---

## 1. Stack

- Python 3
- PyTorch
- Hugging Face Transformers
- Apple Silicon MPS

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`

Use the corresponding Hugging Face tokenizer. Keep V1 deliberately simple. Do not use MLX, llama.cpp, vLLM, FlashAttention, CUDA, quantization, batching, or serving frameworks.

**Dtype:** load the model in `torch.bfloat16` explicitly:

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
```

Do not rely on the library default. Transformers 4.x defaults to fp32 and v5 defaults to `"auto"`, and the choice doubles or halves every KV byte count the lab reports. If bf16 is unsupported on the selected device (older macOS on MPS), fail with a clear error naming the dtype and device. Do not fall back silently.

**Versions:** pin exact versions (`==`) of `torch` and `transformers` in `requirements.txt`, using the versions you tested with. The Transformers cache API has changed several times (`key_cache`/`value_cache` lists, then per-layer `cache.layers[i].keys`/`.values`, and tuple caches removed in v5). Write the cache-inspection code against the pinned version's API only. Do not add compatibility branches for other versions.

## 2. Repository layout and CLI

```
kv-cache-lab/
├── requirements.txt
├── hardware_info.py
├── kv_lab.py
└── README.md
```

Primary commands:

```
python kv_lab.py
python kv_lab.py --prompt "Explain why the sky is blue"
python kv_lab.py --device cpu
python kv_lab.py --decode-steps 20
python kv_lab.py --dtype float32
```

`--dtype` accepts `bfloat16` (default) and `float32`. It exists so the recompute check (section 14a) can be compared across precisions.

Default device is MPS.

## 3. Hardware preflight

`hardware_info.py` exposes a function, e.g. `print_hardware_info(device: str)`, that prints:

- macOS version
- Apple chip/model, if detectable (`sysctl -n machdep.cpu.brand_string`)
- system memory
- Python version
- PyTorch version
- Transformers version
- MPS available: true/false
- MPS built: true/false
- selected device

`kv_lab.py` imports this function and calls it first, passing the device chosen by `--device`. Running `python hardware_info.py` on its own prints the same report with the default device (MPS).

If MPS is unavailable and `--device cpu` was not supplied, fail with a clear error.

## 4. Load and inspect the model

Load SmolLM2-135M-Instruct, call `model.eval()`, and run all inference under `torch.inference_mode()`.

Print:

- number of layers
- hidden size
- attention heads
- KV heads
- head dimension
- vocabulary size
- model dtype

**Head dimension:** use `config.head_dim` if the config defines it, otherwise `hidden_size // num_attention_heads`. Cross-check it against layer 0's projection weights:

```
k_proj.out_features == num_key_value_heads * head_dim
q_proj.out_features == num_attention_heads * head_dim
```

Fail loudly if either check fails.

Determine whether the model uses multi-head attention (MHA), grouped-query attention (GQA), or multi-query attention (MQA) from `num_attention_heads` and `num_key_value_heads`:

- MHA: `kv_heads == attention_heads`
- MQA: `kv_heads == 1`
- GQA: otherwise; also print the group size (`attention_heads / kv_heads` query heads share each KV head)

Use actual model configuration values, not hard-coded numbers.

## 5. Tokenization

Use a small default prompt:

```
The capital of France is
```

Feed the prompt as raw text. **Do not apply the chat template.** The template adds a system prompt of roughly 30 tokens and would bury the prompt tokens the lab is trying to show. Print a one-line note that the chat template is intentionally not applied. The Instruct model still completes a plain-text prompt like this one sensibly.

Print the prompt, total token count, and a row for each token containing token index, token ID, and decoded token/string. Print whether the tokenizer inserted any special tokens (e.g. BOS). Do not dump embeddings or large tensors.

## 6. Instrument the attention layer

Attach PyTorch forward hooks to the first transformer layer's `q_proj`, `k_proj`, and `v_proj` modules. Do not modify Hugging Face or PyTorch source code. Make module discovery reasonably robust if names differ.

For Q, K, and V, log:

- operation name
- output tensor shape (lead with this)
- input tensor shape
- dtype
- device

**Output shapes are the main signal.** The three projections read the same normalized hidden state, so their inputs are identical and show only the sequence dimension. The outputs show both the sequence dimension and the head layout. Under GQA, K and V outputs are narrower than Q:

```
PREFILL                               (sequence dimension = N)
q_proj  out [1, N, attention_heads * head_dim]   in [1, N, hidden_size]
k_proj  out [1, N, kv_heads * head_dim]          in [1, N, hidden_size]
v_proj  out [1, N, kv_heads * head_dim]          in [1, N, hidden_size]

DECODE                                (sequence dimension = 1)
q_proj  out [1, 1, attention_heads * head_dim]   in [1, 1, hidden_size]
k_proj  out [1, 1, kv_heads * head_dim]          in [1, 1, hidden_size]
v_proj  out [1, 1, kv_heads * head_dim]          in [1, 1, hidden_size]
```

Print the real numbers, not the symbolic names. Next to each K/V output shape, print the per-head reshape it corresponds to (`[1, kv_heads, seq, head_dim]`) so the reader can connect it to the cache shape in section 9.

Both contrasts must be obvious in the output: N vs 1 on the sequence dimension, and Q width vs K/V width.

## 7. Warmup

Before measured work, perform one short unmeasured prefill **and one unmeasured decode step** using that warmup cache. Decode calls have a different shape from prefill, so without a decode warmup the first measured decode step pays one-time setup cost and looks slower than the rest.

Discard the warmup cache. Do not mix warmup output with the real experiment. Disable hook logging during warmup.

## 8. Prefill experiment

```
==================================================
PREFILL
==================================================
[monotonic timestamp] PREFILL_START
```

Run the entire prompt through the model in one invocation with `use_cache=True`. Do not use `generate()`.

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=True,
)
```

Measure latency correctly. MPS work can execute asynchronously, so synchronize before and after timing when supported:

```python
sync()                        # torch.mps.synchronize() on MPS, no-op on CPU
start = time.perf_counter()
outputs = model(...)
sync()
end = time.perf_counter()
```

Print: prompt token count, new tokens processed in this call, prefill latency, and prefill throughput in tokens/sec. Performance is secondary to correctness (see section 17 on how to read the timings).

## 9. Inspect the initial KV cache

Retrieve `outputs.past_key_values` and inspect it using the pinned Transformers version's cache API (section 1).

For layer 0 print:

- K shape
- V shape
- dtype
- sequence length
- number of elements
- bytes

Then print the meaning of each dimension using real model values:

```
K shape: [batch, kv_heads, sequence_length, head_dim]
V shape: [batch, kv_heads, sequence_length, head_dim]
```

Point out that the cache stores `kv_heads` heads, not `attention_heads` heads. This is where GQA saves memory.

## 10. Compute KV-cache memory

Calculate KV-cache memory across all transformer layers from the actual tensors. Do not estimate from a formula when tensors are available.

```
KV bytes = sum(numel * element_size) over all K and V tensors in all layers
```

Print per-layer KV bytes, total KV bytes/MiB, sequence length, and especially:

```
bytes_per_cached_token = total_KV_bytes / cached_token_count
```

As a sanity check, also print the formula value and assert that it matches the measured value:

```
expected = num_layers * 2 * kv_heads * head_dim * dtype_bytes
```

For SmolLM2-135M in bf16, expect this to be about 23 KB per token (30 layers, 3 KV heads, head_dim 64, from memory of the config). Verify against the real config.

## 11. Manual decode loop

Do not call `model.generate()`. Implement autoregressive generation manually:

1. Take the logits for the final prompt token from the prefill output.
2. Select the first generated token with greedy argmax.
3. Feed only that one token into the model, passing the KV cache from the previous invocation.
4. Retrieve the updated KV cache and select the next token from the new logits with greedy argmax.
5. Repeat steps 3 and 4 for `--decode-steps` iterations (default 10).

**Token accounting.** With `--decode-steps N`:

- The model is called N times in decode, each with exactly one input token.
- N + 1 tokens are generated in total: one chosen from the prefill logits, and one chosen from each decode call's logits.
- The KV cache ends at `prompt_tokens + N`. The last generated token was never fed back, so it has no K/V entries.

Print this accounting in the summary so the off-by-one is explained, not hidden.

## 12. Decode-step output

```
==================================================
DECODE STEP 1
==================================================
Input tokens this model invocation: 1
Input token ID:   <id>
Input token text: " Paris"
KV length before: N
[time] DECODE_01_START
```

Call the model with exactly one new token plus the previous KV cache. Pass `attention_mask`, `cache_position`, and `position_ids` as the pinned Transformers version requires. If an attention mask is passed, it must cover the cached tokens plus the new one (length `N + step`).

```
[time] DECODE_01_END
Output token ID:   <id>
Output token text: "."
KV length after:  N+1
KV growth:        +1 token
Keys attended:    N+1
Decode latency:   X ms
KV cache size:    Y MiB
```

"Keys attended" is the cache length after the append: the new token's query attends to every cached key, including its own. Print it so the reader sees that decode computes Q/K/V for one token but attention still reads the whole cache.

The progression should make the state transition unmistakable:

```
Prefill:  KV 0   -> N
Decode 1: KV N   -> N+1
Decode 2: KV N+1 -> N+2
Decode 3: KV N+2 -> N+3
```

## 13. Keep Q/K/V hooks enabled during decode

This is a core requirement. The hooks from section 6 stay active through every decode step, and the output visibly contrasts prefill (sequence dimension N) with decode (sequence dimension 1) on the Q, K, and V output shapes.

Do not imply that K/V for previous tokens are recomputed during ordinary decode. The existing K/V entries are reused; only the new token creates new K and V entries.

## 14. Show generated text

After the manual loop, print the original prompt, the list of N + 1 generated tokens (ID and text), and the full resulting text. This verifies that manual generation worked end-to-end.

## 14a. Recompute check

Test the claim that a token's K/V never change once computed. After the decode loop, run every cached position (prompt plus all fed-back tokens) through the model in one uncached call, and compare its K/V with the cache built incrementally.

Print, per layer, the max absolute K and V difference, split into prompt positions and decode positions. Then print the fraction of bit-identical elements, the overall max difference, and whether greedy argmax over the full pass reproduces every generated token. For each token that differs, print both choices and the full pass's top-2 logit gap.

The check reports what it finds. Do not assert equality: caching is exact in real arithmetic, but floating-point results depend on call shape, so the two paths need not agree bit for bit.

## 15. Final experiment summary

```
==================================================
SUMMARY
==================================================
Model: SmolLM2-135M-Instruct  (bf16, mps)
Attention: GQA, <attention_heads> query heads / <kv_heads> KV heads
Prompt tokens: N

PREFILL
  new input tokens:  N
  KV before:         0
  KV after:          N
  KV memory:         X MiB

DECODE
  decode calls:          10
  new tokens per call:   1

KV progression:
  0    -> N       prefill
  N    -> N+1     decode 1
  N+1  -> N+2     decode 2
  ...
  N+9  -> N+10    decode 10

Generated tokens:      11  (1 from prefill logits + 10 from decode logits)
Final KV length:       N+10  (last generated token not yet fed back)
KV bytes/token:        X bytes (measured), X bytes (formula)
```

Also print a short educational summary:

- Prefill runs the whole prompt in one call and creates K/V entries for every prompt token.
- Each decode call processes one new token and appends one token's worth of K/V per layer.
- Decode avoids recomputing old K/V, but attention in every decode step still reads the entire cache. Per-step cost therefore grows with context length. V2 measures this.

## 16. Timestamp markers for profiling

```
12345.123456 PREFILL_START
12345.145331 PREFILL_END

12345.146002 DECODE_01_START
12345.152982 DECODE_01_END

12345.153401 DECODE_02_START
12345.160238 DECODE_02_END
```

Use `time.perf_counter()` or another monotonic high-resolution clock. These markers will later help correlate model-level events with Apple Instruments / Metal System Trace. Do not implement Instruments integration in V1.

## 17. README requirements

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python hardware_info.py
python kv_lab.py
```

The README should briefly explain:

- what prefill is
- what decode is
- what a KV cache is
- why the cache grows by one token per ordinary autoregressive decode step
- that decode still reads the whole cache on every step
- how to read the timings: at 135M parameters and a few prompt tokens, Python and kernel-launch overhead dominate, so prefill and decode latencies will be close and "prefill tokens/sec" says little about the hardware. The timings in V1 exist to validate the measurement harness. V2 makes them meaningful by growing the context.

## 18. Implementation philosophy

This code is for someone coming from a systems-programming background. Prefer explicit control flow over abstractions that hide inference state.

```python
outputs = model(...)
cache = outputs.past_key_values
log_cache(cache)
next_token = ...
outputs = model(next_token, past_key_values=cache)
```

`kv_lab.py` should be readable top to bottom. Comments should explain why something is done, not basic Python syntax.

## 19. V1 non-goals

- vLLM
- paged attention
- FlashAttention analysis
- continuous batching
- prefix caching
- KV-cache quantization
- speculative decoding
- multi-user batching
- multi-GPU
- CUDA
- Nsight
- MLX
- llama.cpp
- model parallelism
- tensor parallelism
- production serving
- chat templates

## 20. Acceptance criteria

Running `python kv_lab.py` should make the following structural facts visible:

| Observation | PREFILL | DECODE |
|---|---|---|
| New input tokens | N | 1 |
| New Q computed | N tokens | 1 token |
| New K computed | N tokens | 1 token |
| New V computed | N tokens | 1 token |
| Q vs K/V output width | `attention_heads * head_dim` vs `kv_heads * head_dim` | same |
| Existing KV | None | Reused |
| KV length before | 0 | N (first step) |
| KV length after | N | N+1 (first step) |
| Keys attended | N | N+1, then N+2, ... |

The measured KV bytes/token must equal the formula value from section 10.

The experiment succeeds if I can concretely see that prefill processes the full prompt and constructs the initial KV cache, and that decode then sends one new token through the model at a time, reuses all prior keys and values, attends over the full cache, and appends exactly one token's worth of KV state per step.

## Deferred to V2

Do not add the memory-bandwidth experiment yet. V2 will hold the model fixed, increase context length, and compare fixed model-weight bytes against linearly growing KV-cache bytes. This will connect the basic mechanics to the memory-bandwidth limits of long-context inference.

### Notes for V2

These were checked against the pinned versions (torch 2.14.0, transformers 5.17.0) during V1 review.

- **DynamicCache copies itself on every append.** `DynamicLayer.update` grows K/V with `torch.cat` (`transformers/cache_utils.py`, lines 146-147), which allocates new tensors and copies the whole cache for every layer on every decode step. Per-step memory traffic is then roughly weights + a read and a write of the full cache, not weights + one read of the cache, so timings will grow faster than (weight bytes + KV bytes) / bandwidth. Use a preallocated `StaticCache` for the clean bandwidth model, or measure both and report the difference.
- **GQA heads are not expanded at the Transformers level.** With an all-ones attention mask, Transformers passes `attention_mask=None` to SDPA and calls it with `enable_gqa=True`; `repeat_kv` was never called in prefill or decode on MPS (30 of 30 layers). The V2 sweep indicates PyTorch's MPS SDPA kernel does not expand the 3 KV heads internally either: DynamicCache measured about 2.7 passes over the cache per step, matching 3 (the `cat` read and write plus one attention read); internal expansion would have shown about 9. Passing a real padding mask would change this path.
- **Timing already brackets a device sync.** `timed_call` calls `torch.mps.synchronize()` before and after the model call, and the `.item()` on argmax runs after the end timestamp. Keep that structure when timing longer contexts.
- **Tied embeddings.** SmolLM2 has `tie_word_embeddings: true`, so every decode step reads the full 49,152 × 576 embedding matrix for the output projection. Count it in the weight bytes read per step. For an untied model, the input embedding table is read one row per token.

---

# KV Cache Lab V2: Decode cost vs context length

**Goal:** Measure how decode latency grows as the KV cache grows, on the same model and machine as V1, and compare the growth with the bytes each decode step has to move. V1 showed the mechanics. V2 answers whether a decode step gets slower in proportion to the extra bytes.

## V2.1 Scope

- Same model (`HuggingFaceTB/SmolLM2-135M-Instruct`), same dtype (bf16), same pinned versions, same device (MPS, with `--device cpu` supported).
- New script `kv_bandwidth.py`. Do not change `kv_lab.py`; V1's output is published. Reuse V1 helpers (`sync`, `kv_bytes_per_layer`, `mib`) by importing them rather than copying.
- The model was trained on up to 8,192 tokens (`max_position_embeddings`). V2 goes past that on purpose: generated text beyond 8,192 tokens is meaningless, but the bytes read are real, and bytes are what V2 measures. Print this once when a length above the limit is used.

## V2.2 CLI

```
python kv_bandwidth.py
python kv_bandwidth.py --lengths 512,2048,8192,32768
python kv_bandwidth.py --steps 100
python kv_bandwidth.py --cache dynamic
python kv_bandwidth.py --out results.csv
```

Defaults: `--lengths 512,1024,2048,4096,8192,16384,32768`, `--steps 50`, `--cache both` (`dynamic`, `static`, or `both`), `--out results.csv`.

## V2.3 Measure the machine's bandwidth

Before any model work, measure achieved GPU memory bandwidth with a large on-device copy (for example a 1 GiB bf16 tensor, `y.copy_(x)`, warmed up, then timed over 20 iterations with a sync before and after). Report GB/s counting both the read and the write. Print it next to Apple's quoted figure (273 GB/s) and use the measured value in every calculation.

## V2.4 Per-length measurement

For each cache type and each context length L:

1. Create a fresh cache. For `static`, preallocate `max_cache_len = L + warmup + steps`.
2. Fill it with one prefill of L tokens. Token content does not matter; use a seeded random sequence so runs repeat. Do not time the prefill.
3. Run 5 unmeasured decode steps.
4. Time `--steps` decode steps individually, with a sync before and after each, as in V1's `timed_call`.
5. Record length, cache type, KV bytes at the start of the timed steps (from the real tensors), and the median, p10 and p90 step latency.

Also record whether the KV heads were expanded before attention at that length (see V2.6).

Count filled cache slots as `L + warmup steps` rather than calling `get_seq_length()`: with the pinned version, `StaticCache.get_seq_length()` returns a tensor equal to the preallocated capacity, not the filled length.

After all lengths, re-run the first length for the first cache type and report the drift from its first measurement, as a thermal check. Print a warning if the medians differ by more than 5%.

## V2.5 Output

Print a table per cache type:

```
context   KV MiB   median ms   p10 ms   p90 ms   KV expanded
    512     11.2        8.06     7.9      8.3    no
    ...
```

Write the same rows to the CSV.

Then fit a straight line of median latency against KV bytes, per cache type, and print:

- intercept (ms): the cost of a decode step with an empty cache, which is mostly fixed per-call overhead plus reading the weights;
- slope, expressed as effective GB/s: KV bytes added per extra millisecond;
- passes over the cache per step: measured bandwidth divided by effective GB/s. About 1 means each step reads the cache once. Larger values mean each step moves the cache more than once (for example, DynamicCache's copy on every append).

Print the fit's R² so a poor fit is visible.

## V2.6 Explain the StaticCache result

A probe during V1 review found StaticCache slower than DynamicCache on MPS (32k context: 31.9 ms vs 18.2 ms). Before trusting the comparison, find out why, and print the reason in the output. Candidates to check:

- whether Transformers passes an attention mask to SDPA for StaticCache, which disables the `enable_gqa` path and triggers `repeat_kv` (3 KV heads copied to 9 before attention);
- whether attention runs over the full preallocated length rather than the filled length.

Detect expansion without modifying library source, for example by wrapping `transformers.integrations.sdpa_attention.repeat_kv` with a counter at runtime. Keep that wrapper small and clearly marked as instrumentation.

## V2.7 Acceptance criteria

- Decode latency at each length is reported with median and spread, for both cache types.
- The measured copy bandwidth is printed and used.
- The line fit gives an intercept, an effective bandwidth and passes per step, with R².
- The StaticCache vs DynamicCache difference has an explanation backed by the V2.6 check, or the output says plainly that it is unexplained.
- The thermal drift check is reported.

## V2.8 Non-goals

Other models, MLX, `torch.compile`, quantization, batching, chunked or paged attention, and any change to `kv_lab.py`.

## V2.9 Blog chart

The CSV feeds an interactive chart in part 2 of the blog post, built as a Hugo shortcode like part 1's slider. It plots latency against context length for both cache types. A slider picks a context length and shows its KV size, latency, and the bytes-per-step estimate. The data is embedded in the shortcode, and there are no external scripts.

---

# KV Cache Lab V3: Moving a KV cache vs rebuilding it

**Goal:** Measure, on one datacenter GPU, how long it takes to rebuild a KV cache with prefill versus bring an existing cache back from somewhere else (another GPU buffer, host memory, local disk, network storage), across context lengths, and find the break-even for each place. This is the trade-off behind KV offload, prefix caching and disaggregated inference: ship the cache or recompute it.

## V3.1 Scope

- New script `kv_offload.py`. Reuse helpers from `kv_bandwidth.py` (`sync`, bandwidth benchmark) by importing them. Do not change V1 or V2 behaviour.
- Device: CUDA only (H100). Runs on Modal through `modal_run.py`, or directly on an SSH-accessible box.
- Two models:
  - `HuggingFaceTB/SmolLM2-135M-Instruct`, for continuity with parts 1 and 2. Prefill is cheap here, so recompute is expected to win at most lengths.
  - One realistic 7B/8B-class model with GQA that does not need a gated licence, e.g. `Qwen/Qwen2.5-7B-Instruct`. Read layers, KV heads and head_dim from the config at runtime and measure bytes per token from real tensors, as in V1; do not hard-code them. Expected about 57 KB per token in bf16 (28 layers × 2 × 4 KV heads × 128 × 2 bytes), to be verified.
- Context lengths: 1k, 4k, 16k, 32k, capped at each model's trained context (`max_position_embeddings`). Unlike V2, stay within the trained context, because this experiment decodes after reloading and checks the output.
- bf16, batch 1, greedy, pinned library versions.

## V3.2 What to time

For each model and context length L:

1. **Recompute:** prefill L tokens into a fresh cache. This is the cost of rebuilding. Median of 5 runs after 1 warmup.
2. **Offload:** copy the filled cache's K/V tensors (all layers) from GPU to each tier. Report time and GB/s. Allocation happens before timing.
3. **Reload:** copy the cache back from each tier into preallocated GPU tensors, ready for decode. Report time and GB/s.

Tiers:

| Tier | Offload | Reload |
|---|---|---|
| GPU → GPU (another buffer in HBM) | device copy | device copy |
| Host RAM, pinned | `copy_` to pinned CPU tensors | `copy_` back, non_blocking + sync |
| Host RAM, pageable | `copy_` to ordinary CPU tensors | `copy_` back |
| Local disk | write one file per run (raw bytes), fsync | read file into pinned memory, then copy to GPU |
| Network storage (Modal Volume or NFS, if available) | as local disk | as local disk |

For disk tiers, report whether the read could have come from the OS page cache. On Modal we cannot drop caches, so the reported disk reload may be a warm read; say so in the output. On a box with root, drop caches (`echo 3 > /proc/sys/vm/drop_caches`) before each cold read and report both cold and warm.

Every timed region brackets a `torch.cuda.synchronize()`.

## V3.3 Correctness

A reloaded cache must be the same bytes. For each tier, after reloading, decode 20 greedy tokens from the reloaded cache and from the original cache, and check that tokens match and logits are bit-identical. This is the baseline part 2's update recommended for testing offload systems: diff against a normal cached run, not a recompute.

## V3.4 Output

Per model, a table:

```
context  KV MiB  recompute ms  GPU ms  pinned ms  pageable ms  disk ms  net ms   (reload)
```

plus the same for offload, the measured GB/s per tier, and the H100's copy bandwidth from V2's benchmark.

Then the break-even per tier: the smallest context length at which reload is faster than recompute, or "reload always faster" / "recompute always faster" within the tested range. Also print the ratio recompute / reload at the largest context.

Write all rows to a CSV.

## V3.5 Network prediction (labelled as a prediction)

Using the measured bytes per token, print the predicted time to move each context's cache over named links, for comparison with recompute: 100 Gb/s and 400 Gb/s Ethernet/InfiniBand, and PCIe host-to-device as measured. Label these clearly as `bytes / nominal link speed` estimates, not measurements. This is the number disaggregated inference pays to move a cache from a prefill machine to a decode machine.

## V3.6 Known pitfalls

- cuDNN attention on H100 has a large per-new-shape cost (found during V2 follow-up: DynamicCache decode 62 ms vs 14 ms with cuDNN SDPA disabled). Prefill uses a fixed shape per length, so it is unaffected, but the correctness decode steps are not. Disable cuDNN SDPA for the decode checks, and record the setting.
- Give the container real CPU cores (8) and enough ephemeral disk for the largest cache; a 32k cache on a 7B model is about 1.8 GB.
- First run downloads the 7B model (~15 GB) into the cache volume; do not time it.

## V3.7 Non-goals

Real cross-machine transfer (RDMA, NIXL), GPUDirect Storage, cache compression or quantization, vLLM/SGLang, prefix-matching logic, and batching. V3.5 predicts network cost; measuring it needs two machines.

---

# KV Cache Lab V4: Inference disaggregation, part 2 (moving the cache between GPUs)

**Goal:** On one machine with two NVLink-connected H100s, measure what it costs to run prefill on GPU 0 and decode on GPU 1 instead of both on GPU 0: how long the KV cache transfer takes, how much it delays the first decode step, and how much of it can be hidden by sending the cache layer by layer while prefill is still running.

Part 3 predicted this cost from link speeds; V4 measures it on a real GPU-to-GPU link.

## V4.1 Scope

- New script `kv_disagg.py`. Reuse helpers from `kv_offload.py` and `kv_bandwidth.py` by importing them. Do not change earlier scripts' behaviour.
- Hardware: Modal `gpu="H100:2"` via `modal_run.py`. Print `nvidia-smi topo -m` and `torch.cuda.can_device_access_peer(0, 1)` so the post can state whether the two GPUs talk over NVLink or PCIe. If they are PCIe-only, stop and report rather than publishing numbers as NVLink.
- Model: `Qwen/Qwen2.5-7B-Instruct`, one copy loaded on each GPU. Context lengths 1k, 4k, 16k, 32k. bf16, batch 1, greedy.
- cuDNN attention disabled for decode steps (see V3.6).

## V4.2 Link bandwidth

Before the model work, measure GPU 0 → GPU 1 copy bandwidth with a 1 GiB tensor, as V2 did for on-device copies. Report GB/s. This is the number to compare with Part 3's pinned host RAM (about 27 GB/s) and the nominal network links.

## V4.3 What to time

For each context length L, median of 5 runs after 1 warmup, with syncs on both GPUs around each timed region:

1. **Colocated:** prefill L tokens on GPU 0, then one decode step on GPU 0. Report prefill time and time until the first decode step finishes.
2. **Disaggregated, bulk:** prefill on GPU 0, then copy the whole cache (all layers' K/V) to preallocated tensors on GPU 1, then one decode step on GPU 1. Report the transfer time on its own and the time until the first decode step finishes.
3. **Disaggregated, layer by layer:** prefill on GPU 0 with a forward hook on each decoder layer that, as soon as the layer finishes, starts copying that layer's K/V to GPU 1 on a separate CUDA stream. Then one decode step on GPU 1 once all copies are done. Report the exposed transfer time: (time until all copies finish) minus (prefill time alone).

The headline numbers per length are prefill time, bulk transfer time, exposed layer-by-layer transfer time, and each as a percentage of prefill.

## V4.4 Correctness

Decode 20 greedy tokens colocated on GPU 0 and disaggregated on GPU 1 (from both transfer methods). Tokens must match. Report whether logits are bit-identical; different GPUs of the same model should give identical results, but if they do not, report the max difference rather than failing.

## V4.5 Output

A table per context length (prefill, bulk transfer, layer-wise exposed transfer, overhead %), the measured GPU-to-GPU bandwidth, the topology printout, and the correctness results. Write rows to a CSV. For comparison, also print Part 3's predicted network transfer times for the same cache sizes (bytes / nominal link speed, labelled as predictions).

## V4.6 Non-goals

Interference between prefill and decode on a shared GPU (a later post), attention/FFN disaggregation, cross-machine transfer, NIXL/NCCL-based transfer libraries, vLLM/SGLang, batching.

---

# KV Cache Lab V5: Inference disaggregation, part 1 (why split: interference)

**Goal:** Show the problem disaggregation exists to solve. One user is decoding on a GPU; a new request with a long prompt arrives and its prefill runs on the same GPU. Measure how long the decoding user waits between tokens, compare with chunked prefill (the usual fix without a second GPU), and with prefill on a separate GPU. V4 (part 2) then measures what that separation costs.

## V5.1 Scope

- New script `kv_interference.py`. Reuse helpers from `kv_disagg.py`, `kv_offload.py` and `kv_bandwidth.py`. Do not change earlier scripts' behaviour.
- Hardware and model as V4: Modal `H100:2` with NVLink, `Qwen/Qwen2.5-7B-Instruct` loaded on each GPU, bf16, greedy, cuDNN attention off for decode steps.
- Scheduling is modelled the way serving engines do it, one iteration at a time: each iteration runs either a decode step for user A or a piece of prefill work for request B. There is no true simultaneous sharing of one GPU; that is out of scope.

## V5.2 Scenario

- User A has a 1,024-token prompt, already prefilled, and decodes 80 tokens.
- Request B arrives just before A's 20th decode step, with a prompt of L tokens, L in 4k, 16k and 32k (one run per L).
- Record, for every decode step of A, the wall time since A's previous token, with a sync on both GPUs so each gap includes all work the scheduler ran in between. Also record B's time to first token: from arrival to B's first generated token being available.

## V5.3 Modes

1. **Colocated:** B's whole prefill runs on GPU 0 between two of A's decode steps, also on GPU 0. A's gap at that step includes all of B's prefill.
2. **Chunked prefill:** B's prompt is fed in chunks of C tokens (C = 512 and 2,048), each chunk a forward call that appends to B's cache. The scheduler alternates one chunk of B with one decode step of A until B's prefill is done, all on GPU 0.
3. **Disaggregated, thread:** A decodes on GPU 1. B's prefill runs on GPU 0 in a separate thread of the same process, then B's cache is copied to GPU 1 (bulk, as V4). A's decode loop does not wait for B on the GPU, but both threads share one Python interpreter lock.
4. **Disaggregated, process:** as above, but B's prefill runs in a separate worker process with its own copy of the model on GPU 0 (spawned with `torch.multiprocessing`, fed prompts over a queue). B's cache stays on GPU 0; V4 measures the copy. B's time to first token uses the worker's `perf_counter` timestamp, which is comparable across processes on Linux (CLOCK_MONOTONIC).

Finding from the first full run: the thread version stalled A for most of B's prefill (1,161 ms at 32k) even though the GPUs are separate; the process version did not (worst gap 26 ms). Keep both modes so the difference is reproducible.

## V5.4 Correctness

A's 80 generated tokens must be identical across all modes: the scheduling changes when A's steps run, not what they compute. For chunked mode, also check that B's first token matches B's first token from a single unchunked prefill.

## V5.5 Output

Per mode and L: A's median gap, A's worst gap, the number of A's gaps above 2× the median, and B's time to first token. Write every one of A's gaps to a CSV (mode, L, step, gap_ms) so the post can chart gap over time: a spike for colocated, a run of smaller bumps for chunked, flat for disaggregated.

## V5.6 Non-goals

Batched decode of many users, B decoding after its first token, true concurrent execution of prefill and decode on one GPU (streams, MPS, time-slicing), vLLM/SGLang, and cross-machine transfer.
