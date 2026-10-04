"""How much error FP4 rounding adds to real weights, with and without block scales (SPEC.md V11).

    python kv_fp4_error.py                                   # default model and matrix
    python kv_fp4_error.py --model HuggingFaceTB/SmolLM2-135M-Instruct --tensor model.layers.15.mlp.down_proj.weight

Runs on a laptop CPU. Downloads only the safetensors shard that holds the chosen matrix, then
quantizes that matrix, and every other 2-D linear weight in the same shard, four ways:
  direct  round each weight to the nearest FP4 value, no scale
  tensor  one scale for the whole matrix (largest weight maps to 6)
  mxfp4   one power-of-two scale per 32 weights (OCP MX spec)
  nvfp4   one FP8 (E4M3) scale per 16 weights, plus one FP32 scale per matrix (NVIDIA)
Writes results_fp4_error.csv.
"""
import argparse
import csv
import json
import pathlib

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # E2M1 magnitudes; sign is separate
OUT = pathlib.Path(__file__).parent / "results_fp4_error.csv"


def round_fp4(x):
    # Nearest FP4 value, keeping the sign; anything past 6 saturates to 6.
    mag = x.abs().clamp(max=6.0)
    idx = (mag.unsqueeze(-1) - FP4).abs().argmin(-1)
    return FP4[idx] * x.sign()


def direct(w, amax):
    return round_fp4(w)


def tensor_scale(w, amax):
    s = amax / 6
    return round_fp4(w / s) * s


def mxfp4(w, amax):
    b = w.reshape(-1, 32)
    amax = b.abs().amax(1, keepdim=True).clamp(min=1e-30)
    # Shared exponent = floor(log2(amax)) - 2, where 2 is FP4's largest exponent.
    s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
    return (round_fp4(b / s) * s).reshape(w.shape)


def nvfp4(w, amax):
    b = w.reshape(-1, 16)
    g = amax / (6 * 448)  # per-matrix FP32 scale, so block scales fit E4M3 (max 448)
    sb = (b.abs().amax(1, keepdim=True) / (6 * g)).to(torch.float8_e4m3fn).to(torch.float32)
    s = (sb * g).clamp(min=1e-30)
    return (round_fp4(b / s) * s).reshape(w.shape)


METHODS = {"direct": direct, "tensor": tensor_scale, "mxfp4": mxfp4, "nvfp4": nvfp4}


def measure(w):
    a = w.abs()
    sample = a.flatten() if w.numel() < 2**24 else a.flatten()[::7]  # torch.quantile caps its input at 2^24
    row = {"weights": w.numel(), "median_abs": a.median().item(), "p99_abs": sample.quantile(0.99).item(),
           "max_abs": a.max().item()}
    for name, f in METHODS.items():
        # Chunks keep the argmin's memory small. Per-matrix scales use the whole matrix's max.
        q = torch.cat([f(c, a.max()) for c in w.split(256)])
        row[f"{name}_zeroed_pct"] = round(100 * ((q == 0) & (w != 0)).float().mean().item(), 2)
        # Average error relative to the average weight size.
        row[f"{name}_err_pct"] = round(100 * ((q - w).abs().mean() / a.mean()).item(), 2)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--tensor", default="model.layers.18.mlp.down_proj.weight")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    try:
        index = json.load(open(hf_hub_download(args.model, "model.safetensors.index.json")))
        shard = index["weight_map"][args.tensor]
    except Exception:
        shard = "model.safetensors"  # single-file models have no index
    path = hf_hub_download(args.model, shard)

    rows = []
    with safe_open(path, "pt") as f:
        names = [args.tensor] + sorted(n for n in f.keys() if n != args.tensor)
        for n in names:
            t = f.get_tensor(n)
            if t.ndim != 2 or "embed" in n or "lm_head" in n or t.shape[1] % 32:
                continue
            row = {"model": args.model, "tensor": n, "headline": n == args.tensor, "dtype": str(t.dtype).split(".")[-1],
                   "shape": "x".join(map(str, t.shape)), **measure(t.float())}
            rows.append(row)
            print(json.dumps(row), flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} matrices -> {args.out}")


if __name__ == "__main__":
    main()
