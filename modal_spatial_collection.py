"""Four B200 workers collecting diverse snapshots from the frozen game player."""
from pathlib import Path
import json
import modal
from modal_policy import image as base_image, ROOT

app = modal.App("generals-spatial-reader-data")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image
         .add_local_file(ROOT / "modal_spatial_collection.py", "/project/modal_spatial_collection.py")
         .add_local_file(ROOT / "runs/play-2000/L_7d_gae90_ema_2000.eqx", "/project/checkpoint.eqx"))


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=14400,
              volumes={"/experiments": volume}, retries=0, max_containers=4,
              scaledown_window=2, single_use_containers=True, include_source=False)
def collect_worker(dataset: str, worker: int):
    import os
    import re
    import shutil
    import subprocess
    import sys
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", dataset) or worker not in range(4):
        raise ValueError("Invalid collection request")
    output = Path("/experiments") / dataset / f"worker-{worker}"
    if output.exists():
        raise FileExistsError(output)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "B200" not in hardware:
        raise RuntimeError(f"Expected a B200, received {hardware}")
    command = [sys.executable, "-u", "/project/scripts/collect_spatial_dataset.py",
               "--checkpoint", "/project/checkpoint.eqx", "--output", str(output),
               "--split", "train", "--worker", str(worker), "--batches", "8"]
    if worker in (0, 1):
        command.extend(["--extra-split", "validation" if worker == 0 else "test"])
    print(json.dumps({"worker": worker, "hardware": hardware, "command": command}), flush=True)
    log_path = Path(f"/tmp/spatial-collection-{worker}.log")
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd="/project", stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       env={**os.environ, "JAX_COMPILATION_CACHE_DIR": "/tmp/jax-cache-spatial"})
            for line in process.stdout:
                print(f"worker-{worker}: {line}", end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith('{"stage": "shard_complete"'):
                    shutil.copy2(log_path, output / "collection.log")
                    volume.commit()
            code = process.wait()
        if code:
            raise RuntimeError(f"Collection worker {worker} exited with status {code}")
        record = json.loads((output / "manifest.json").read_text())
        record["hardware"] = hardware
        return record
    finally:
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, output / "collection.log")
        volume.commit()


@app.function(image=image, cpu=1, memory=2048, timeout=300,
              volumes={"/experiments": volume}, include_source=False)
def assemble(dataset: str):
    volume.reload()
    root = Path("/experiments") / dataset
    records = [json.loads((root / f"worker-{worker}/manifest.json").read_text()) for worker in range(4)]
    hashes = {r["checkpoint_sha256"] for r in records}
    if len(hashes) != 1 or not all(r["player_checkpoint_unchanged"] for r in records):
        raise ValueError("Frozen checkpoint mismatch")
    result = {"name": dataset, "checkpoint_sha256": hashes.pop(), "config": records[0]["config"],
              "collection": {"envs_per_batch": 64, "steps": 1024, "stride": 32,
                             "pool_size": 6272, "gpu_workers": 4, "gpu": "B200"}}
    for split, expected in (("train", 131072), ("validation", 8192), ("test", 8192)):
        shards = []
        for worker, record in enumerate(records):
            for shard in record["shards"]:
                if shard["split"] == split:
                    shards.append({**shard, "path": f"worker-{worker}/{shard['path']}"})
        count = sum(s["samples"] for s in shards)
        if count != expected:
            raise ValueError(f"Wrong {split} size: {count}, expected {expected}")
        result[split] = {"samples": count, "shards": shards,
                         "games": sum(s["games"] for s in shards),
                         "unique_maps": len({m for s in shards for m in s["map_ids"]})}
    sets = [{m for s in result[split]["shards"] for m in s["map_ids"]}
            for split in ("train", "validation", "test")]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Map overlap across dataset splits")
    result["map_splits_verified_disjoint"] = True
    (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, cpu=.25, memory=2048, timeout=28800,
              include_source=False, retries=0, max_containers=1)
def coordinate(dataset: str):
    list(collect_worker.starmap([(dataset, worker) for worker in range(4)]))
    result = assemble.remote(dataset)
    summary = {"dataset": dataset, **{split: result[split]["samples"]
                                     for split in ("train", "validation", "test")}}
    print(json.dumps({"stage": "dataset_complete", **summary}), flush=True)
    return summary


@app.local_entrypoint()
def main(dataset: str = "iter2000-spatial-data-v3"):
    call = coordinate.spawn(dataset)
    print(json.dumps({"coordinator_call_id": call.object_id, "dataset": dataset}), flush=True)
