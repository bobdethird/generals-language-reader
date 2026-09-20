"""Repeat the local iteration-2000 reader pilot on one H100 for timing comparison.

Only the selected dataset, frozen checkpoint, and reader source are uploaded.
The game-training and checkpoint-evaluation applications are independent.
"""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "runs/iter2000-reader-data-v2"
CHECKPOINT = ROOT / "runs/play-2000/L_7d_gae90_ema_2000.eqx"
MODEL = {"model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "transformers==4.57.6", "huggingface-hub==0.36.2",
                 "safetensors==0.8.0", "numpy==2.5.3")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', "
        "cache_dir='/project/.cache/huggingface/hub')\"")
    .env({"PYTHONPATH": "/project", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "reader", "/project/reader", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "scripts/train_behavior_reader.py", "/project/scripts/train_behavior_reader.py")
    .add_local_file(ROOT / "reader-model.json", "/project/reader-model.json")
    .add_local_file(ROOT / "modal_reader_pilot.py", "/project/modal_reader_pilot.py")
    .add_local_dir(DATA, "/dataset")
    .add_local_file(CHECKPOINT, "/project/checkpoint.eqx")
)
app = modal.App("generals-reader-pilot")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)


@app.function(image=image, gpu="H100", cpu=6, memory=32768, timeout=1800,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True, include_source=False)
def benchmark(name: str):
    import datetime as dt
    import hashlib
    import re
    import shutil
    import subprocess
    import sys
    import time
    import torch

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Invalid experiment name")
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU is required")
    output = Path("/experiments") / name
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads(Path("/dataset/manifest.json").read_text())
    checkpoint = Path("/project/checkpoint.eqx")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source_manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash differs from the local pilot")
    shutil.copytree("/dataset", "/work/data")
    source_manifest["original_checkpoint_path"] = source_manifest["checkpoint"]
    source_manifest["checkpoint"] = str(checkpoint)
    Path("/work/data/manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    command = [sys.executable, "-u", "/project/scripts/train_behavior_reader.py",
               "--data", "/work/data", "--output", str(output), "--device", "cuda",
               "--fresh-adapter", "--warmup-steps", "300", "--predictor-steps", "400",
               "--rl-steps", "40", "--bootstrap-texts", "128", "--eval-samples", "96",
               "--max-tokens", "80", "--seed", "2000144"]
    started = time.monotonic()
    metadata = {"hardware": hardware, "command": command,
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "precision": "float32", "attention": "eager",
                "dataset_manifest_sha256": hashlib.sha256(Path("/dataset/manifest.json").read_bytes()).hexdigest()}
    print(json.dumps(metadata), flush=True)
    try:
        with Path("/work/training.log").open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        metadata.update(exit_code=result, process_seconds=time.monotonic() - started)
        if result:
            raise RuntimeError(f"Reader pilot exited with status {result}")
        return {"hardware": metadata, "report": json.loads((output / "report.json").read_text())}
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if Path("/work/training.log").exists():
            shutil.copy2("/work/training.log", output / "training.log")
        volume.commit()


@app.local_entrypoint()
def main(name: str = "iter2000-reader-h100-v1"):
    if json.loads((ROOT / "reader-model.json").read_text()) != MODEL:
        raise ValueError("Modal language model must match the local pinned model")
    destination = ROOT / "runs" / name
    destination.mkdir(parents=True, exist_ok=False)
    result = benchmark.remote(name)
    (destination / "report.json").write_text(json.dumps(result["report"], indent=2) + "\n")
    (destination / "hardware.json").write_text(json.dumps(result["hardware"], indent=2) + "\n")
    print(json.dumps({"output": str(destination), "hardware": result["hardware"]["hardware"],
                      "seconds": result["report"]["seconds"]}), flush=True)
