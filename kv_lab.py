"""KV Cache Lab V1: make prefill, decode, and KV-cache growth visible.

Reads top to bottom: preflight, load model, tokenize, warmup, prefill,
inspect the cache, manual decode loop, summary.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hardware_info import print_hardware_info

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_PROMPT = "The capital of France is"

# Filled by the forward hooks on layer 0's q/k/v projections, printed after each timed call.
qkv_records = []


def banner(title):
    print("=" * 50)
    print(title)
    print("=" * 50)


def sync(device):
    # MPS kernels run asynchronously. Without a sync, perf_counter() would stop
    # when the work is queued, not when it has finished.
    if device == "mps":
        torch.mps.synchronize()


def find_qkv_module(model):
    # named_modules() walks in registration order, so the first module that owns
    # all three projections is the first transformer layer's attention block.
    for name, module in model.named_modules():
        if all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj")):
            return name, module
    raise SystemExit("Could not find a module with q_proj, k_proj and v_proj children.")


def make_hook(op_name):
    def hook(module, inputs, output):
        # Record only metadata. Reading .shape/.dtype does not force a device sync,
        # so the hooks do not disturb the timing.
        qkv_records.append((op_name, tuple(inputs[0].shape), tuple(output.shape), output.dtype, output.device))
    return hook


def print_qkv_records(heads_by_op, head_dim):
    for op_name, in_shape, out_shape, dtype, device in qkv_records:
        batch, seq, _ = out_shape
        heads = heads_by_op[op_name]
        print(
            f"  {op_name}  out {list(out_shape)!s:<14} in {list(in_shape)!s:<14}"
            f" = [{batch}, {heads} heads, {seq} tok, {head_dim}]  {dtype} {device}"
        )


def timed_call(model, device, **kwargs):
    # Hook output is buffered and printed after the call so terminal I/O
    # stays out of the measured interval.
    qkv_records.clear()
    sync(device)
    start = time.perf_counter()
    outputs = model(**kwargs, use_cache=True)
    sync(device)
    end = time.perf_counter()
    return outputs, start, end


def kv_bytes_per_layer(cache):
    return [
        layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    ]


def mib(num_bytes):
    return num_bytes / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--decode-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    device = args.device
    dtype = getattr(torch, args.dtype)

    # ---- Preflight -------------------------------------------------------
    print_hardware_info(device)
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("ERROR: MPS is not available on this machine. Re-run with --device cpu.")
    try:
        torch.ones(1, dtype=dtype, device=device) + 1
    except (RuntimeError, TypeError) as e:
        raise SystemExit(f"ERROR: {dtype} is not supported on device '{device}': {e}")

    # ---- Load and inspect the model ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # dtype is explicit: library defaults differ across versions and would change every KV byte count.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or hidden_size // num_heads

    attn_name, attn = find_qkv_module(model)
    if attn.q_proj.out_features != num_heads * head_dim:
        raise SystemExit(f"q_proj.out_features={attn.q_proj.out_features} != {num_heads} * {head_dim}")
    if attn.k_proj.out_features != num_kv_heads * head_dim:
        raise SystemExit(f"k_proj.out_features={attn.k_proj.out_features} != {num_kv_heads} * {head_dim}")

    if num_kv_heads == num_heads:
        attention_type = "MHA (multi-head attention)"
    elif num_kv_heads == 1:
        attention_type = "MQA (multi-query attention)"
    else:
        attention_type = (
            f"GQA (grouped-query attention), {num_heads // num_kv_heads} query heads share each KV head"
        )

    banner("MODEL")
    print(f"Model:            {MODEL_ID}")
    print(f"Layers:           {num_layers}")
    print(f"Hidden size:      {hidden_size}")
    print(f"Attention heads:  {num_heads}")
    print(f"KV heads:         {num_kv_heads}")
    print(f"Head dimension:   {head_dim}  (checked against q_proj/k_proj widths)")
    print(f"Vocabulary size:  {config.vocab_size}")
    print(f"Model dtype:      {model.dtype}")
    print(f"Attention type:   {attention_type}")
    print()

    # ---- Tokenization --------------------------------------------------------
    # Raw prompt on purpose: the chat template would add ~30 system-prompt tokens.
    enc = tokenizer(args.prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    prompt_ids = input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    special = [t for t in prompt_ids if t in tokenizer.all_special_ids]

    banner("TOKENIZATION")
    print(f"Prompt: {args.prompt!r}")
    print("Chat template: not applied (raw text prompt, intentional)")
    print(f"Special tokens inserted: {special or 'none'}")
    print(f"Token count: {prompt_len}")
    print(f"  {'idx':>3}  {'id':>6}  text")
    for i, token_id in enumerate(prompt_ids):
        print(f"  {i:>3}  {token_id:>6}  {tokenizer.decode(token_id)!r}")
    print()

    with torch.inference_mode():
        # ---- Warmup ----------------------------------------------------------
        # One prefill and one decode step: the two call shapes hit different
        # kernels, and each pays one-time setup cost on first use. Hooks are
        # not registered yet, so warmup produces no Q/K/V output.
        warm = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        warm_next = warm.logits[:, -1:].argmax(dim=-1)
        warm_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim=1)
        model(input_ids=warm_next, attention_mask=warm_mask, past_key_values=warm.past_key_values, use_cache=True)
        sync(device)
        del warm
        print("Warmup done (1 prefill + 1 decode, unmeasured, cache discarded).\n")

        heads_by_op = {"q_proj": num_heads, "k_proj": num_kv_heads, "v_proj": num_kv_heads}
        for op_name in heads_by_op:
            getattr(attn, op_name).register_forward_hook(make_hook(op_name))

        # ---- Prefill ---------------------------------------------------------
        banner("PREFILL")
        outputs, start, end = timed_call(model, device, input_ids=input_ids, attention_mask=attention_mask)
        cache = outputs.past_key_values
        prefill_ms = (end - start) * 1000

        print(f"{start:.6f} PREFILL_START")
        print(f"Q/K/V projections in {attn_name} (sequence dimension = {prompt_len}):")
        print_qkv_records(heads_by_op, head_dim)
        print(f"{end:.6f} PREFILL_END")
        print(f"Prompt tokens:                     {prompt_len}")
        print(f"New tokens processed in this call: {input_ids.shape[1]}")
        print(f"Keys attended by last prompt token: {cache.get_seq_length()}")
        print(f"Prefill latency:                   {prefill_ms:.2f} ms")
        print(f"Prefill throughput:                {prompt_len / (end - start):.1f} tokens/sec")
        print()

        # ---- Inspect the initial KV cache ------------------------------------
        banner(f"INITIAL KV CACHE (layer 0 of {num_layers})")
        layer0 = cache.layers[0]
        k, v = layer0.keys, layer0.values
        print(f"Cache type:       {type(cache).__name__}")
        print(f"K shape:          {list(k.shape)}")
        print(f"V shape:          {list(v.shape)}")
        print(f"dtype:            {k.dtype}")
        print(f"Sequence length:  {k.shape[2]}")
        print(f"Elements (K+V):   {k.numel() + v.numel()}")
        print(f"Bytes (K+V):      {k.numel() * k.element_size() + v.numel() * v.element_size()}")
        print("Dimensions:")
        print(f"  [0] batch           = {k.shape[0]}")
        print(f"  [1] kv_heads        = {k.shape[1]}  (not {num_heads}: only KV heads are cached, which is the GQA saving)")
        print(f"  [2] sequence_length = {k.shape[2]}  (one entry per cached token)")
        print(f"  [3] head_dim        = {k.shape[3]}")
        print()

        # ---- KV-cache memory -------------------------------------------------
        banner("KV CACHE MEMORY (all layers, from real tensors)")
        per_layer = kv_bytes_per_layer(cache)
        for i, layer_bytes in enumerate(per_layer):
            print(f"  layer {i:>2}: {layer_bytes} bytes")
        total_bytes = sum(per_layer)
        cached_tokens = cache.get_seq_length()
        bytes_per_token = total_bytes // cached_tokens
        expected_bytes_per_token = num_layers * 2 * num_kv_heads * head_dim * k.element_size()
        print(f"Total KV bytes:          {total_bytes} ({mib(total_bytes):.3f} MiB)")
        print(f"Cached tokens:           {cached_tokens}")
        print(f"bytes_per_cached_token:  {bytes_per_token}")
        print(
            f"Formula check:           {num_layers} layers * 2 (K,V) * {num_kv_heads} kv_heads"
            f" * {head_dim} head_dim * {k.element_size()} bytes = {expected_bytes_per_token}"
        )
        assert bytes_per_token == expected_bytes_per_token, "measured KV bytes/token does not match formula"
        print("Measured matches formula.")
        print()

        # ---- Manual decode loop ----------------------------------------------
        # The prefill logits already choose the first generated token. Each
        # decode call feeds the most recent token and chooses the next one.
        generated = [outputs.logits[0, -1].argmax().item()]
        progression = [(0, prompt_len, "prefill")]

        for step in range(1, args.decode_steps + 1):
            token_id = generated[-1]
            # DynamicCache is updated in place, so read the length before the call.
            kv_before = cache.get_seq_length()
            # The mask covers cached tokens plus the new one. The pinned
            # Transformers version derives cache_position and position_ids
            # from the cache length, so they are not passed.
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim=1)
            step_input = torch.tensor([[token_id]], device=device)

            banner(f"DECODE STEP {step}")
            print(f"Input tokens this model invocation: {step_input.shape[1]}")
            print(f"Input token ID:   {token_id}")
            print(f"Input token text: {tokenizer.decode(token_id)!r}")
            print(f"KV length before: {kv_before}")

            outputs, start, end = timed_call(
                model, device, input_ids=step_input, attention_mask=attention_mask, past_key_values=cache
            )
            cache = outputs.past_key_values
            kv_after = cache.get_seq_length()
            generated.append(outputs.logits[0, -1].argmax().item())
            progression.append((kv_before, kv_after, f"decode {step}"))

            print(f"{start:.6f} DECODE_{step:02d}_START")
            print("Q/K/V projections (sequence dimension = 1, old K/V are reused from the cache):")
            print_qkv_records(heads_by_op, head_dim)
            print(f"{end:.6f} DECODE_{step:02d}_END")
            print(f"Output token ID:   {generated[-1]}")
            print(f"Output token text: {tokenizer.decode(generated[-1])!r}")
            print(f"KV length after:  {kv_after}")
            print(f"KV growth:        +{kv_after - kv_before} token")
            print(f"Keys attended:    {kv_after}  (the new query reads every cached key, including its own)")
            print(f"Decode latency:   {(end - start) * 1000:.2f} ms")
            print(f"KV cache size:    {mib(sum(kv_bytes_per_layer(cache))):.3f} MiB")
            print()

        # ---- Recompute check -------------------------------------------------
        # Claim under test: a token's K/V never change once computed. Rebuild
        # every cached position in one uncached pass over the same tokens and
        # compare it with the cache that decode built one token at a time.
        fed_ids = prompt_ids + generated[:-1]
        fresh = model(input_ids=torch.tensor([fed_ids], device=device), use_cache=True)
        fresh_cache = fresh.past_key_values

        banner("RECOMPUTE CHECK (incremental cache vs one full pass)")
        print(f"Recomputed {len(fed_ids)} positions in one call. Max |difference| per layer:")
        print(f"  {'layer':>5}  {'K prompt':>9}  {'K decode':>9}  {'V prompt':>9}  {'V decode':>9}")
        equal_elements = total_elements = 0
        worst = {"prompt": 0.0, "decode": 0.0}
        for i, (cached, recomputed) in enumerate(zip(cache.layers, fresh_cache.layers)):
            row = []
            for a, b in ((cached.keys, recomputed.keys), (cached.values, recomputed.values)):
                diff = (a.float() - b.float()).abs()
                equal_elements += (diff == 0).sum().item()
                total_elements += diff.numel()
                for part, d in (("prompt", diff[:, :, :prompt_len]), ("decode", diff[:, :, prompt_len:])):
                    worst[part] = max(worst[part], d.max().item())
                    row.append(d.max().item())
            print(f"  {i:>5}  " + "  ".join(f"{x:>9.2e}" for x in row))

        # Does the drift matter? Re-derive every generated token from the full pass.
        recomputed_tokens = fresh.logits[0, prompt_len - 1 :].argmax(dim=-1).tolist()
        largest_k = max(layer.keys.abs().max().item() for layer in cache.layers)
        print(f"Bit-identical elements:    {equal_elements} / {total_elements} ({100 * equal_elements / total_elements:.1f}%)")
        print(f"Max |diff|, prompt positions: {worst['prompt']:.2e}")
        print(f"Max |diff|, decode positions: {worst['decode']:.2e}  (largest |K| in cache: {largest_k:.1f})")
        print(f"Greedy tokens from the full pass match decode: {recomputed_tokens == generated}")
        for i, (ours, theirs) in enumerate(zip(generated, recomputed_tokens)):
            if ours != theirs:
                top2 = fresh.logits[0, prompt_len - 1 + i].float().topk(2).values.tolist()
                print(
                    f"  generated token {i + 1}: decode chose {tokenizer.decode(ours)!r}, full pass chose"
                    f" {tokenizer.decode(theirs)!r} (full-pass top-2 logit gap {top2[0] - top2[1]:.4f})"
                )
        print()

    # ---- Generated text ------------------------------------------------------
    banner("GENERATED TEXT")
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated tokens ({len(generated)}):")
    for i, token_id in enumerate(generated):
        source = "prefill logits" if i == 0 else f"decode {i} logits"
        print(f"  {i + 1:>2}  {token_id:>6}  {tokenizer.decode(token_id)!r:<16} from {source}")
    print(f"Full text: {tokenizer.decode(prompt_ids + generated)!r}")
    print()

    # ---- Summary -------------------------------------------------------------
    final_kv = cache.get_seq_length()
    banner("SUMMARY")
    print(f"Model: {MODEL_ID.split('/')[-1]}  ({args.dtype}, {device})")
    print(f"Attention: {attention_type.split(' ')[0]}, {num_heads} query heads / {num_kv_heads} KV heads")
    print(f"Prompt tokens: {prompt_len}")
    print()
    print("PREFILL")
    print(f"  new input tokens:  {prompt_len}")
    print("  KV before:         0")
    print(f"  KV after:          {prompt_len}")
    print(f"  KV memory:         {mib(total_bytes):.3f} MiB")
    print()
    print("DECODE")
    print(f"  decode calls:          {args.decode_steps}")
    print("  new tokens per call:   1")
    print()
    print("KV progression:")
    for before, after, label in progression:
        print(f"  {before:>4} -> {after:<4}  {label}")
    print()
    print(f"Generated tokens:  {len(generated)}  (1 from prefill logits + {args.decode_steps} from decode logits)")
    print(f"Final KV length:   {final_kv}  (last generated token not yet fed back)")
    print(f"KV bytes/token:    {bytes_per_token} bytes (measured), {expected_bytes_per_token} bytes (formula)")
    print()
    print("What happened:")
    print(f"  - Prefill ran all {prompt_len} prompt tokens in one call and created their K/V entries in every layer.")
    print("  - Each decode call processed one new token and appended one token's worth of K/V per layer.")
    print("  - Decode never recomputed old K/V, but every decode step's attention still read the")
    print("    whole cache, so per-step cost grows with context length. V2 measures this.")


if __name__ == "__main__":
    main()
