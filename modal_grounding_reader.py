"""Eight additional grounding passes for the frozen iteration-2000 language reader."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
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
    .add_local_file(ROOT / "scripts/train_grounding_reader.py", "/project/scripts/train_grounding_reader.py")
    .add_local_file(ROOT / "reader-model.json", "/project/reader-model.json")
    .add_local_file(ROOT / "modal_grounding_reader.py", "/project/modal_grounding_reader.py")
    .add_local_dir(ROOT / "runs/iter2000-reader-data-v2", "/dataset")
    .add_local_dir(ROOT / "runs/iter2000-grounding-audit-v1", "/audit")
    .add_local_file(ROOT / "runs/play-2000/L_7d_gae90_ema_2000.eqx", "/project/checkpoint.eqx")
    .add_local_file(ROOT / "runs/iter2000-reader-mps-v1/before_rl.pt", "/project/initial_reader.pt")
)
app = modal.App("generals-reader-grounding")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)


@app.function(image=image, gpu="H100", cpu=6, memory=32768, timeout=7200,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True, include_source=False)
def train(name: str):
    import datetime as dt
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
    Path("/work").mkdir(exist_ok=True)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    command = [sys.executable, "-u", "/project/scripts/train_grounding_reader.py",
               "--data", "/dataset", "--test-data", "/audit",
               "--initial-reader", "/project/initial_reader.pt",
               "--player-checkpoint", "/project/checkpoint.eqx", "--output", str(output),
               "--device", "cuda", "--epochs", "8", "--batch-size", "16", "--eval-batch-size", "16",
               "--validation-samples", "128", "--test-samples", "256", "--seed", "2000844"]
    started = time.monotonic()
    metadata = {"hardware": hardware, "command": command,
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "precision": "float32", "attention": "eager"}
    print(json.dumps(metadata), flush=True)
    try:
        with Path("/work/training.log").open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith('{"stage": "validation"'):
                    (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    shutil.copy2("/work/training.log", output / "training.log")
                    volume.commit()
            result = process.wait()
        metadata.update(exit_code=result, process_seconds=time.monotonic() - started)
        if result:
            raise RuntimeError(f"Grounding experiment exited with status {result}")
        return {"hardware": metadata, "report": json.loads((output / "report.json").read_text())}
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if Path("/work/training.log").exists():
            shutil.copy2("/work/training.log", output / "training.log")
        volume.commit()


@app.local_entrypoint()
def main(name: str = "iter2000-grounding-h100-v1"):
    if json.loads((ROOT / "reader-model.json").read_text()) != MODEL:
        raise ValueError("Modal language model must match the local pinned model")
    destination = ROOT / "runs" / name
    destination.mkdir(parents=True, exist_ok=False)
    result = train.remote(name)
    (destination / "report.json").write_text(json.dumps(result["report"], indent=2) + "\n")
    (destination / "hardware.json").write_text(json.dumps(result["hardware"], indent=2) + "\n")
    print(json.dumps({"output": str(destination), "hardware": result["hardware"]["hardware"],
                      "seconds": result["report"]["seconds"], "best_epoch": result["report"]["best_epoch"],
                      "ready_for_rl": result["report"]["ready_for_rl"]}), flush=True)
