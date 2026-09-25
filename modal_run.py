"""Run a lab script on a Modal GPU and save its output locally.

    modal run modal_run.py                                   # kv_bandwidth.py, four-cache sweep on an H100
    modal run modal_run.py --script kv_offload.py --args "--net-dir /net" --out results_offload.csv
    modal run modal_run.py --script kv_disagg.py --args "" --out results_disagg.csv   # on two H100s
    modal run modal_run.py --script kv_interference.py --args "" --out results_interference.csv   # two H100s

The printed report goes to stdout and the CSV to --out (default results_h100.csv).
"""

import pathlib
import shlex

import modal

GPU = "H100"
LAB_FILES = ["kv_bandwidth.py", "kv_offload.py", "kv_disagg.py", "kv_interference.py", "kv_lab.py", "hardware_info.py"]

image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements("requirements.txt")
for name in LAB_FILES:
    image = image.add_local_file(name, f"/lab/{name}")

# The models are downloaded once and kept here, so repeat runs skip the download.
hf_cache = modal.Volume.from_name("kv-cache-lab-hf", create_if_missing=True)
# Scratch space on Modal's network storage, for kv_offload.py's optional "net" tier.
scratch = modal.Volume.from_name("kv-cache-lab-scratch", create_if_missing=True)
app = modal.App("kv-cache-lab")


# Decode at this model size is limited by Python launching GPU kernels, so give the
# container real CPU cores; the default fraction of a core makes timings noisy.
# kv_offload.py keeps several copies of a 7B model's cache in host memory, hence 64 GiB.
FUNCTION_OPTIONS = dict(cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache, "/net": scratch}, timeout=3600)


@app.function(gpu=GPU, **FUNCTION_OPTIONS)
def run(script: str, args: str) -> tuple[str, str]:
    return execute(script, args)


# kv_disagg.py and kv_interference.py need two GPUs on one machine.
TWO_GPU_SCRIPTS = ("kv_disagg.py", "kv_interference.py")


@app.function(gpu=f"{GPU}:2", **FUNCTION_OPTIONS)
def run_two_gpus(script: str, args: str) -> tuple[str, str]:
    return execute(script, args)


def execute(script, args):
    import os
    import subprocess

    command = ["python", script, *shlex.split(args), "--out", "/tmp/results.csv"]
    if script == "kv_bandwidth.py":
        command[2:2] = ["--device", "cuda"]
    result = subprocess.run(
        command, cwd="/lab", env={**os.environ, "HF_HOME": "/hf"}, capture_output=True, text=True
    )
    hf_cache.commit()
    csv_path = pathlib.Path("/tmp/results.csv")
    return result.stdout + result.stderr, csv_path.read_text() if csv_path.exists() else ""


@app.local_entrypoint()
def main(script: str = "kv_bandwidth.py", args: str = "--cache all", out: str = "results_h100.csv"):
    runner = run_two_gpus if script in TWO_GPU_SCRIPTS else run
    report, csv_text = runner.remote(script, args)
    print(report)
    if csv_text:
        pathlib.Path(out).write_text(csv_text)
        print(f"Saved CSV to {out}")
    else:
        print("No CSV produced; see the report above.")
