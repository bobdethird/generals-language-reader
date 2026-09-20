"""Train an independent observer on one B200, then audit completed reader text."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT

app = modal.App("generals-decision-context-audit")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image
         .add_local_file(ROOT / "scripts/train_decision_listener.py", "/project/scripts/train_decision_listener.py")
         .add_local_file(ROOT / "modal_decision_listener.py", "/project/modal_decision_listener.py"))


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def fit(dataset: str, name: str):
    import re
    import shutil
    import subprocess
    import sys
    import torch
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", v) for v in (dataset, name)):
        raise ValueError("Invalid name")
    volume.reload()
    output = Path("/experiments") / name
    if output.exists():
        raise FileExistsError(output)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "B200" not in hardware or not torch.cuda.is_available():
        raise RuntimeError("Requested B200 unavailable")
    command = [sys.executable, "-u", "/project/scripts/train_decision_listener.py",
               "--data", str(Path("/experiments") / dataset), "--output", str(output)]
    print(json.dumps({"hardware": hardware, "command": command}), flush=True)
    log_path = Path("/tmp/listener-training.log")
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith('{"stage": "validation"'):
                    shutil.copy2(log_path, output / "training.log")
                    volume.commit()
            code = process.wait()
        if code:
            raise RuntimeError(f"Observer fit failed with status {code}")
        return {"output": name, "hardware": hardware}
    finally:
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, output / "training.log")
        (output / "hardware.json").write_text(json.dumps({"hardware": hardware})+'\n')
        volume.commit()


@app.function(image=image, cpu=8, memory=16384, timeout=3600,
              volumes={"/experiments": volume}, retries=0, max_containers=1, include_source=False)
def audit(dataset: str, name: str, reader_comparison: str):
    import subprocess
    import sys
    volume.reload()
    root = Path("/experiments")
    command = [sys.executable, "-u", "/project/scripts/train_decision_listener.py",
               "--data", str(root/dataset), "--output", str(root/name), "--audit", "--readers",
               str(root/f"{reader_comparison}-spatial"), str(root/f"{reader_comparison}-policy_aux")]
    subprocess.run(command, cwd="/project", check=True)
    volume.commit()
    return json.loads((root/name/"generated-context-audit.json").read_text())


@app.function(image=image, cpu=.25, memory=2048, timeout=28800,
              include_source=False, retries=0, max_containers=1)
def coordinate(dataset: str, name: str, reader_comparison: str, reader_call_id: str):
    fitted = fit.remote(dataset, name)
    print(json.dumps({"stage": "observer_fit_complete", **fitted}), flush=True)
    result = modal.FunctionCall.from_id(reader_call_id).get()
    if result["name"] != reader_comparison or result["dataset"] != dataset:
        raise ValueError("Reader comparison mismatch")
    return audit.remote(dataset, name, reader_comparison)


@app.local_entrypoint()
def main(dataset: str = "iter2000-spatial-data-v3", name: str = "iter2000-decision-listener-v1",
         reader_comparison: str = "iter2000-decision-comparison-v2",
         reader_call_id: str = "fc-01M2YPZJE67NBJPFHKB2RTERMA"):
    call = coordinate.spawn(dataset, name, reader_comparison, reader_call_id)
    print(json.dumps({"coordinator_call_id": call.object_id, "dataset": dataset, "name": name}), flush=True)
