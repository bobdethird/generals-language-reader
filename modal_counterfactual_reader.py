"""Checkpoint-3800 input-intervention collection, independent of the game learner."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("generals-3800-counterfactual-reader")
volume = modal.Volume.from_name("generals-reader-experiments")
players = modal.Volume.from_name("generals-policy-checkpoints")
SOURCE = ("published-20260920-3am-eight-b200-8gpu-train/checkpoints/"
          "L_7d_gae90/L_7d_gae90_ema_3800.eqx")
if modal.is_local():
    from modal_policy import image as base
    image = base.add_local_file(ROOT/"modal_counterfactual_reader.py", "/project/modal_counterfactual_reader.py")
else:
    image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=7200,
              volumes={"/experiments": volume, "/players": players.with_mount_options(read_only=True)},
              retries=0, max_containers=4, single_use_containers=True, scaledown_window=2, include_source=False)
def pilot(name="iter3800-counterfactual-pilot-v1"):
    import hashlib
    import re
    import shutil
    import subprocess
    import sys
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid output name")
    volume.reload(); players.reload()
    root = Path("/experiments")/name
    root.mkdir(exist_ok=False)
    checkpoint = root/"player-3800.eqx"
    shutil.copy2(Path("/players")/SOURCE, checkpoint)
    provenance = {"iteration": 3800, "player_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "source": SOURCE, "hardware": subprocess.check_output(
                      ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()}
    (root/"provenance.json").write_text(json.dumps(provenance, indent=2)+'\n'); volume.commit()
    commands = [
        [sys.executable, "-u", "/project/scripts/collect_spatial_dataset.py", "--checkpoint", str(checkpoint),
         "--output", str(root/"snapshots"), "--split", "test", "--worker", "0", "--batches", "1",
         "--envs", "32", "--steps", "512", "--stride", "32", "--seed-base", "840000000"],
        [sys.executable, "-u", "/project/scripts/collect_counterfactuals.py", "--checkpoint", str(checkpoint),
         "--source", str(root/"snapshots/test-w0-b0"), "--output", str(root/"evidence"), "--count", "256"],
    ]
    try:
        with (root/"execution.log").open("w") as log:
            for command in commands:
                p = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
                for line in p.stdout:
                    print(line, end="", flush=True); log.write(line); log.flush()
                if p.wait():
                    raise RuntimeError(f"Pilot subprocess failed: {p.returncode}")
        report = json.loads((root/"evidence/report.json").read_text())
        print(json.dumps({"stage": "pilot_complete", "report": report}), flush=True)
        return report
    finally:
        volume.commit()


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=3600,
              volumes={"/experiments":volume},retries=0,max_containers=1,
              single_use_containers=True,scaledown_window=2,include_source=False)
def reprobe():
    from scripts.collect_counterfactuals import extract
    volume.reload()
    root=Path("/experiments/iter3800-counterfactual-pilot-v1")
    try:
        return extract(root/"player-3800.eqx",root/"snapshots/test-w0-b0",root/"evidence-bf16-input",256)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(name: str = "iter3800-counterfactual-pilot-v1", reuse_snapshots: bool = False):
    call = reprobe.spawn() if reuse_snapshots else pilot.spawn(name)
    print(json.dumps({"call_id": call.object_id, "name": name}), flush=True)
