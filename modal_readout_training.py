"""Train one B200 language reader conditioned on the player's frozen action head."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT

app = modal.App("generals-frozen-readout-language")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image
         .add_local_file(ROOT/"scripts/train_decision_reader.py", "/project/scripts/train_decision_reader.py")
         .add_local_file(ROOT/"runs/iter2000-policy-head-b200.pt", "/project/policy-head.pt")
         .add_local_file(ROOT/"modal_readout_training.py", "/project/modal_readout_training.py"))


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def train():
    import subprocess
    import sys
    import shutil
    volume.reload()
    output = Path("/experiments/iter2000-decision-readout-v2")
    if output.exists():
        raise FileExistsError(output)
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "B200" not in hardware:
        raise RuntimeError("Expected B200")
    command = [sys.executable, "-u", "/project/scripts/train_decision_reader.py",
               "--data", "/experiments/iter2000-spatial-data-v3",
               "--initial-reader", "/experiments/iter2000-spatial-comparison-v1-spatial/best.pt",
               "--player-checkpoint", "/project/checkpoint.eqx", "--policy-head", "/project/policy-head.pt",
               "--output", str(output), "--variant", "readout", "--epochs", "4", "--batch-size", "128"]
    print(json.dumps({"hardware": hardware, "command": command}), flush=True)
    logpath = Path("/tmp/readout-training.log")
    try:
        with logpath.open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith('{"stage": "validation"'):
                    shutil.copy2(logpath, output/"training.log")
                    volume.commit()
            code = process.wait()
        if code:
            raise RuntimeError(f"Readout reader failed with status {code}")
        report = json.loads((output/"report.json").read_text())
        return {"output": output.name, "best_step": report["best_step"], "test": report["test"],
                "head_parity": report["frozen_head_test"]}
    finally:
        output.mkdir(exist_ok=True)
        shutil.copy2(logpath, output/"training.log")
        (output/"hardware.json").write_text(json.dumps({"hardware": hardware, "command": command}, indent=2)+'\n')
        volume.commit()


@app.local_entrypoint()
def main():
    call = train.spawn()
    print(json.dumps({"call_id": call.object_id}), flush=True)
