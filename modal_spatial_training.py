"""Two B200 comparison runs using the completed spatial snapshot dataset."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("generals-spatial-reader-training")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
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
    .add_local_file(ROOT / "scripts/train_spatial_reader.py", "/project/scripts/train_spatial_reader.py")
    .add_local_file(ROOT / "reader-model.json", "/project/reader-model.json")
    .add_local_file(ROOT / "modal_spatial_training.py", "/project/modal_spatial_training.py")
    .add_local_file(ROOT / "runs/play-2000/L_7d_gae90_ema_2000.eqx", "/project/checkpoint.eqx")
)


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=2,
              scaledown_window=2, single_use_containers=True, include_source=False)
def train_variant(dataset: str, name: str, adapter: str):
    import re
    import shutil
    import subprocess
    import sys
    import time
    import torch
    for value in (dataset, name):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
            raise ValueError("Invalid path component")
    if adapter not in ("tokenwise", "spatial"):
        raise ValueError("Invalid adapter")
    volume.reload()
    data = Path("/experiments") / dataset
    output = Path("/experiments") / name
    if not (data / "manifest.json").is_file():
        raise ValueError("Dataset has not finished collection and split verification")
    if output.exists():
        raise FileExistsError(output)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "B200" not in hardware or not torch.cuda.is_available():
        raise RuntimeError(f"Expected a B200, received {hardware}")
    command = [sys.executable, "-u", "/project/scripts/train_spatial_reader.py",
               "--data", str(data), "--initial-reader", "/experiments/iter2000-grounding-h100-v1/best.pt",
               "--player-checkpoint", "/project/checkpoint.eqx", "--output", str(output),
               "--adapter", adapter, "--epochs", "4", "--batch-size", "128",
               "--eval-batch-size", "64", "--eval-every", "512"]
    metadata = {"hardware": hardware, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "adapter": adapter, "command": command, "started_unix": time.time()}
    print(json.dumps(metadata), flush=True)
    log_path = Path("/tmp") / f"{name}.log"
    started = time.monotonic()
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(f"{adapter}: {line}", end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith('{"stage": "validation"'):
                    (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    shutil.copy2(log_path, output / "training.log")
                    volume.commit()
            code = process.wait()
        metadata.update(exit_code=code, seconds=time.monotonic() - started)
        if code:
            raise RuntimeError(f"{adapter} training failed with status {code}")
        return {"hardware": metadata, "report": json.loads((output / "report.json").read_text())}
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copy2(log_path, output / "training.log")
        volume.commit()


@app.function(image=image, cpu=.25, memory=2048, timeout=28800,
              volumes={"/experiments": volume}, include_source=False, retries=0, max_containers=1)
def coordinate(dataset: str, name: str, collection_call_id: str = ""):
    if collection_call_id:
        print(json.dumps({"stage": "waiting_for_dataset", "dataset": dataset,
                          "collection_call_id": collection_call_id}), flush=True)
        result = modal.FunctionCall.from_id(collection_call_id).get()
        if result["dataset"] != dataset:
            raise ValueError("Collection coordinator returned a different dataset")
    names = [(dataset, f"{name}-{kind}", kind) for kind in ("tokenwise", "spatial")]
    summary = {"dataset": dataset, "name": name, "variants": {}}
    for (_, output_name, kind), result in zip(names, train_variant.starmap(names)):
        summary["variants"][kind] = {"output": output_name, "best_step": result["report"]["best_step"],
                                      "test": result["report"]["best_reader_test"]["metrics"]["real"]}
    volume.reload()
    destination = Path("/experiments") / f"{name}-summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    print(json.dumps({"stage": "comparison_complete", **summary}), flush=True)
    return summary


@app.local_entrypoint()
def main(dataset: str = "iter2000-spatial-data-v3", name: str = "iter2000-spatial-comparison-v1",
         collection_call_id: str = ""):
    call = coordinate.spawn(dataset, name, collection_call_id)
    print(json.dumps({"coordinator_call_id": call.object_id, "dataset": dataset, "name": name}), flush=True)
