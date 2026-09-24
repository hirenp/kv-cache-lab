"""Run a lab script on a Modal GPU and save its output locally.

    modal run modal_run.py                                   # kv_bandwidth.py, four-cache sweep on an H100
    modal run modal_run.py --args "--cache static-compiled"  # the compiled StaticCache run
    modal run modal_run.py --script kv_offload.py --args "--net-dir /net" --out results_offload.csv

The printed report goes to stdout and the CSV to --out (default results_h100.csv).
"""

import pathlib
import shlex

import modal

GPU = "H100"
LAB_FILES = ["kv_bandwidth.py", "kv_offload.py", "kv_lab.py", "hardware_info.py"]

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
@app.function(
    gpu=GPU, cpu=8.0, memory=65536, image=image, volumes={"/hf": hf_cache, "/net": scratch}, timeout=3600
)
def run(script: str, args: str) -> tuple[str, str]:
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
    report, csv_text = run.remote(script, args)
    print(report)
    if csv_text:
        pathlib.Path(out).write_text(csv_text)
        print(f"Saved CSV to {out}")
    else:
        print("No CSV produced; see the report above.")
