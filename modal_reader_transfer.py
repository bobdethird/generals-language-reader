"""Adapt the reader to iteration 3800 only after the fresh 2000 audit passes.

The training checkpoint volume is mounted read-only. All new work is written to
the separate reader-experiment volume; no deployed game player is replaced.
"""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent

app = modal.App("generals-reader-transfer-3800")
volume = modal.Volume.from_name("generals-reader-experiments")
players = modal.Volume.from_name("generals-policy-checkpoints")
AUDIT_CALL = "fc-01M2YW6GKRVTYV92D4QAK57BN4"
SOURCE = ("published-20260920-3am-eight-b200-8gpu-train/checkpoints/"
          "L_7d_gae90/L_7d_gae90_ema_3800.eqx")
CAMPAIGN = "iter3800-reader-transfer-v1"
DATASET = "iter3800-spatial-data-v1"
READER = "iter3800-decision-readout-v1"
OBSERVERS = "iter3800-decision-listener-v1"
if modal.is_local():
    from modal_policy import image as collection_base
    from modal_spatial_training import image as language_base
    collection_image = collection_base.add_local_file(
        ROOT / "modal_reader_transfer.py", "/project/modal_reader_transfer.py")
    language_image = (language_base
        .add_local_file(ROOT / "scripts/train_decision_reader.py", "/project/scripts/train_decision_reader.py")
        .add_local_file(ROOT / "scripts/train_decision_listener.py", "/project/scripts/train_decision_listener.py")
        .add_local_file(ROOT / "modal_reader_transfer.py", "/project/modal_reader_transfer.py"))
else:
    # Modal hydrates the already-built images. Importing local image-builder
    # modules here would require unrelated files in both distinct runtimes.
    collection_image = language_image = modal.Image.debian_slim(python_version="3.12")


def gated_reader(report):
    """A failed or incomplete prior audit cannot allocate transfer GPUs."""
    if report.get("positions") != 32768 or not report.get("transfer_pilot_gate", {}).get("passed"):
        return None
    if report.get("selected_reader") != "iter2000-decision-readout-v2":
        raise ValueError("Transfer implementation requires the verified readout architecture")
    if not report.get("reader_checkpoint_sha256"):
        raise ValueError("Missing source-reader identity")
    return report["selected_reader"]


@app.function(image=collection_image, cpu=4, memory=16384,
              timeout=1800, volumes={"/experiments": volume,
                  "/players": players.with_mount_options(read_only=True)},
              retries=0, max_containers=1, include_source=False)
def prepare(prior):
    import hashlib
    import os
    import shutil
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np
    import jax.numpy as jnp
    from scripts.collect_activations import Config, build_network, jax, eqx
    volume.reload(); players.reload()
    root = Path("/experiments")
    previous = root / prior["selected_reader"]
    if hashlib.sha256((previous / "best.pt").read_bytes()).hexdigest() != prior["reader_checkpoint_sha256"]:
        raise ValueError("The selected reader changed after its fresh audit")
    old = json.loads((previous / "report.json").read_text())
    if not old["frozen_player_unchanged"] or not old["frozen_language_unchanged"]:
        raise ValueError("Prior frozen-weight verification failed")
    output = root / CAMPAIGN
    output.mkdir(exist_ok=False)
    checkpoint = output / "L_7d_gae90_ema_3800.eqx"
    shutil.copy2(Path("/players") / SOURCE, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = Config.from_yaml("/project/configs/averagejoe-published.yaml")
    net = eqx.tree_deserialise_leaves(checkpoint, build_network(cfg, jax.random.PRNGKey(cfg.seed)))
    if not net.use_bf16 or net.policy_head.weight.shape != (81, 448):
        raise ValueError("Unexpected iteration-3800 architecture")
    rounded = lambda x: np.asarray(x.astype(jnp.bfloat16).astype(jnp.float32))
    np.savez(output / "policy-head.npz", weight=rounded(net.policy_head.weight),
             bias=rounded(net.policy_head.bias))
    record = {"iteration": 3800, "weights": "ema", "source_volume": "generals-policy-checkpoints",
              "source_path": SOURCE, "player_sha256": sha, "use_bf16": True,
              "initial_reader": prior["selected_reader"],
              "initial_reader_sha256": prior["reader_checkpoint_sha256"],
              "decision_scope": "Frozen player; adapt only the language adapter, then revalidate",
              "passed_iteration_2000_audit": prior}
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    volume.commit()
    return {"player_sha256": sha, "initial_reader": prior["selected_reader"]}


def logged_run(command, output, *, checkpoint_events=False):
    import os
    import subprocess
    output = Path(output)
    # The called script owns output-directory creation.
    logpath = Path("/tmp") / (output.name + ".log")
    try:
        with logpath.open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "JAX_COMPILATION_CACHE_DIR": "/tmp/reader-transfer-jax"})
            for line in process.stdout:
                print(line, end="", flush=True); log.write(line); log.flush()
                if checkpoint_events and line.startswith('{"stage": "validation"'):
                    volume.commit()
            code = process.wait()
        if code:
            raise RuntimeError(f"Reader-transfer subprocess exited with status {code}")
    finally:
        if output.exists() and logpath.exists():
            import shutil
            shutil.copy2(logpath, output / "execution.log")
        volume.commit()


@app.function(image=collection_image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=4,
              single_use_containers=True, scaledown_window=2, include_source=False)
def collect(worker):
    import subprocess
    import sys
    if worker not in range(4):
        raise ValueError("Invalid worker")
    volume.reload()
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    if "B200" not in hardware:
        raise RuntimeError("Expected B200")
    root = Path("/experiments")
    output = root / DATASET / f"worker-{worker}"
    if output.exists():
        raise FileExistsError(output)
    command = [sys.executable, "-u", "/project/scripts/collect_spatial_dataset.py",
               "--checkpoint", str(root / CAMPAIGN / "L_7d_gae90_ema_3800.eqx"),
               "--output", str(output), "--split", "train", "--worker", str(worker),
               "--batches", "8", "--seed-base", "380000000"]
    if worker in (0, 1):
        command += ["--extra-split", "validation" if worker == 0 else "test"]
    logged_run(command, output)
    return json.loads((output / "manifest.json").read_text())


@app.function(image=language_image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def adapt(initial_reader):
    import hashlib
    import numpy as np
    import subprocess
    import sys
    import torch
    volume.reload()
    root = Path("/experiments"); campaign = root / CAMPAIGN
    provenance = json.loads((campaign / "provenance.json").read_text())
    if initial_reader != provenance["initial_reader"]:
        raise ValueError("Unapproved initial reader")
    if hashlib.sha256((root / initial_reader / "best.pt").read_bytes()).hexdigest() != provenance["initial_reader_sha256"]:
        raise ValueError("Initial reader changed")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    if "B200" not in hardware:
        raise RuntimeError("Expected B200")
    with np.load(campaign / "policy-head.npz", allow_pickle=False) as head:
        torch.save({"weight": torch.from_numpy(head["weight"]), "bias": torch.from_numpy(head["bias"]),
                    "player_sha256": provenance["player_sha256"], "use_bf16": True}, campaign / "policy-head.pt")
    command = [sys.executable, "-u", "/project/scripts/train_decision_reader.py",
               "--data", str(root / DATASET), "--initial-reader", str(root / initial_reader / "best.pt"),
               "--player-checkpoint", str(campaign / "L_7d_gae90_ema_3800.eqx"),
               "--policy-head", str(campaign / "policy-head.pt"), "--output", str(root / READER),
               "--variant", "readout", "--epochs", "4", "--batch-size", "128"]
    logged_run(command, root / READER, checkpoint_events=True)
    # The independent observers also need targets from the new player.
    logged_run([sys.executable, "-u", "/project/scripts/train_decision_listener.py",
                "--data", str(root / DATASET), "--output", str(root / OBSERVERS)], root / OBSERVERS)
    subprocess.run([sys.executable, "-u", "/project/scripts/train_decision_listener.py",
                    "--data", str(root / DATASET), "--output", str(root / OBSERVERS),
                    "--audit", "--audit-device", "cuda", "--readers", str(root / READER)],
                   cwd="/project", check=True)
    report = json.loads((root / READER / "report.json").read_text())
    result = {"status": "pilot_complete", "reader": READER, "player_iteration": 3800,
              "test": report["test"]["metrics"]["real"], "head_parity": report["frozen_head_test"],
              "player_unchanged": report["frozen_player_unchanged"],
              "language_unchanged": report["frozen_language_unchanged"],
              "context_audit": json.loads((root / OBSERVERS / "generated-context-audit.json").read_text())}
    (campaign / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=collection_image, cpu=1, memory=4096, timeout=28800,
              volumes={"/experiments": volume}, retries=0, max_containers=1, include_source=False)
def coordinate():
    prior = modal.FunctionCall.from_id(AUDIT_CALL).get()
    initial = gated_reader(prior)
    if initial is None:
        result = {"status": "deferred", "reason": "Iteration-2000 fresh audit did not pass all transfer criteria",
                  "gate": prior.get("transfer_pilot_gate")}
        volume.reload()
        (Path("/experiments") / f"{CAMPAIGN}-deferred.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit(); print(json.dumps(result), flush=True)
        return result
    prepared = prepare.remote(prior)
    records = list(collect.map(range(4)))
    volume.reload()
    root = Path("/experiments")
    manifest = {"name": DATASET, "checkpoint_sha256": prepared["player_sha256"],
                "config": records[0]["config"]}
    for record in records:
        if record["checkpoint_sha256"] != prepared["player_sha256"] or not record["player_checkpoint_unchanged"]:
            raise ValueError("Collection checkpoint mismatch")
    map_sets = []
    for split, expected in (("train", 131072), ("validation", 8192), ("test", 8192)):
        shards = [{**s, "path": f"worker-{w}/{s['path']}"} for w, r in enumerate(records)
                  for s in r["shards"] if s["split"] == split]
        maps = {m for s in shards for m in s["map_ids"]}
        if sum(s["samples"] for s in shards) != expected:
            raise ValueError("Wrong transfer dataset size")
        manifest[split] = {"samples": expected, "shards": shards, "unique_maps": len(maps)}
        map_sets.append(maps)
    old_maps = set()
    for name in ("iter2000-spatial-data-v3", "iter2000-fresh-reader-audit-v1"):
        old = json.loads((root / name / "manifest.json").read_text())
        old_maps.update(m for split in ("train", "validation", "test") if split in old
                        for s in old[split]["shards"] for m in s["map_ids"])
    if any(a & b for i, a in enumerate(map_sets) for b in map_sets[i+1:]) or any(s & old_maps for s in map_sets):
        raise ValueError("Transfer maps overlap training or prior evaluation maps")
    manifest["map_splits_verified_disjoint"] = True
    manifest["disjoint_from_iteration_2000_data"] = True
    (root / DATASET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    return adapt.remote(initial)


@app.local_entrypoint()
def main():
    call = coordinate.spawn()
    print(json.dumps({"call_id": call.object_id, "starts_after": AUDIT_CALL,
                      "condition": "All predeclared transfer gates pass", "campaign": CAMPAIGN}), flush=True)
