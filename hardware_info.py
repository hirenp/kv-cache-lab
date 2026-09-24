"""Hardware and software preflight for KV Cache Lab."""

import platform
import subprocess
import sys

import torch
import transformers


def sysctl(name):
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def print_hardware_info(device):
    memsize = sysctl("hw.memsize")
    memory = f"{int(memsize) / 2**30:.1f} GiB" if memsize else "unknown"

    print("=" * 50)
    print("HARDWARE / SOFTWARE")
    print("=" * 50)
    print(f"macOS version:        {platform.mac_ver()[0] or 'unknown'}")
    print(f"Chip:                 {sysctl('machdep.cpu.brand_string') or 'unknown'}")
    print(f"System memory:        {memory}")
    print(f"Python version:       {platform.python_version()}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"MPS available:        {torch.backends.mps.is_available()}")
    print(f"MPS built:            {torch.backends.mps.is_built()}")
    print(f"Selected device:      {device}")
    print()


if __name__ == "__main__":
    print_hardware_info(sys.argv[1] if len(sys.argv) > 1 else "mps")
