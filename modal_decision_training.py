"""B200 decision-language experiments; player and pretrained decoder stay frozen."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT

app = modal.App("generals-decision-language-reader")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image
         .add_local_file(ROOT / "scripts/train_decision_reader.py", "/project/scripts/train_decision_reader.py")
         .add_local_file(ROOT / "modal_decision_training.py", "/project/modal_decision_training.py"))


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=2,
              scaledown_window=2, single_use_containers=True, include_source=False)
def train_variant(dataset: str, name: str, variant: str):
    import re
    import shutil
    import subprocess
    import sys
    import time
    import torch
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", v) for v in (dataset, name)):
        raise ValueError("Invalid dataset/run name")
    if variant not in ("spatial", "policy_aux", "policy_mask"):
        raise ValueError("Invalid variant")
    volume.reload()
    data, output = Path("/experiments") / dataset, Path("/experiments") / name
    if output.exists():
        raise FileExistsError(output)
    if not (data / "manifest.json").is_file():
        raise ValueError("Dataset not assembled")
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "B200" not in hardware or not torch.cuda.is_available():
        raise RuntimeError("Requested B200 unavailable")
    command = [sys.executable, "-u", "/project/scripts/train_decision_reader.py",
               "--data", str(data), "--initial-reader", "/experiments/iter2000-grounding-h100-v1/best.pt",
               "--player-checkpoint", "/project/checkpoint.eqx", "--output", str(output),
               "--variant", variant, "--epochs", "4", "--batch-size", "128", "--eval-every", "512"]
    metadata = {"hardware": hardware, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "variant": variant, "command": command, "started_unix": time.time()}
    print(json.dumps(metadata), flush=True)
    log_path = Path("/tmp") / f"{name}.log"
    started = time.monotonic()
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(f"{variant}: {line}", end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith('{"stage": "validation"'):
                    (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    shutil.copy2(log_path, output / "training.log")
                    volume.commit()
            code = process.wait()
        metadata.update(exit_code=code, seconds=time.monotonic()-started)
        if code:
            raise RuntimeError(f"Decision reader {variant} failed with status {code}")
        report = json.loads((output / "report.json").read_text())
        return {"variant": variant, "output": name, "best_step": report["best_step"], "test": report["test"]}
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "hardware.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copy2(log_path, output / "training.log")
        volume.commit()


@app.function(image=image, cpu=.25, memory=2048, timeout=28800,
              volumes={"/experiments": volume}, include_source=False, retries=0, max_containers=1)
def coordinate(dataset: str, name: str, collection_call_id: str = "", variants: str = "spatial,policy_aux"):
    if collection_call_id:
        print(json.dumps({"stage": "waiting_for_dataset", "call_id": collection_call_id}), flush=True)
        result = modal.FunctionCall.from_id(collection_call_id).get()
        if result["dataset"] != dataset:
            raise ValueError("Dataset name mismatch")
    kinds = variants.split(",")
    if not kinds or len(kinds) != len(set(kinds)) or any(k not in ("spatial", "policy_aux", "policy_mask") for k in kinds):
        raise ValueError("Invalid comparison variants")
    plans = [(dataset, f"{name}-{variant}", variant) for variant in kinds]
    results = list(train_variant.starmap(plans))
    summary = {"dataset": dataset, "name": name, "variants": results}
    volume.reload()
    (Path("/experiments") / f"{name}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    print(json.dumps({"stage": "comparison_complete", **summary}), flush=True)
    return summary


@app.local_entrypoint()
def main(dataset: str = "iter2000-spatial-data-v3", name: str = "iter2000-decision-comparison-v1",
         collection_call_id: str = "", variants: str = "spatial,policy_aux"):
    call = coordinate.spawn(dataset, name, collection_call_id, variants)
    print(json.dumps({"coordinator_call_id": call.object_id, "dataset": dataset, "name": name}), flush=True)
