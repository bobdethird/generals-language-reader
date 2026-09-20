"""Two B200s collect a fresh, never-trained-on final reader audit population."""
from pathlib import Path
import json
import modal
from modal_spatial_collection import image as base_image, ROOT

app = modal.App("generals-fresh-reader-audit-data")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = base_image.add_local_file(ROOT/"modal_fresh_reader_data.py", "/project/modal_fresh_reader_data.py")
DATASET = "iter2000-fresh-reader-audit-v1"


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=7200,
              volumes={"/experiments": volume}, retries=0, max_containers=2,
              single_use_containers=True, scaledown_window=2, include_source=False)
def collect(worker: int):
    import os
    import subprocess
    import sys
    import shutil
    if worker not in (0, 1):
        raise ValueError("Unexpected worker")
    output=Path("/experiments")/DATASET/f"worker-{worker}"
    hardware=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],text=True).strip()
    if "B200" not in hardware:
        raise RuntimeError("Expected B200")
    command=[sys.executable,"-u","/project/scripts/collect_spatial_dataset.py","--checkpoint","/project/checkpoint.eqx",
             "--output",str(output),"--split","test","--worker",str(worker),"--batches","4","--seed-base","190000000"]
    print(json.dumps({"worker":worker,"hardware":hardware,"command":command}),flush=True)
    logpath=Path(f"/tmp/fresh-collection-{worker}.log")
    try:
        with logpath.open("w") as log:
            process=subprocess.Popen(command,cwd="/project",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,bufsize=1,env={**os.environ,"JAX_COMPILATION_CACHE_DIR":"/tmp/jax-fresh-cache"})
            for line in process.stdout:
                print(f"worker-{worker}: {line}",end="",flush=True);log.write(line);log.flush()
                if line.startswith('{"stage": "shard_complete"'):
                    shutil.copy2(logpath,output/"collection.log");volume.commit()
            code=process.wait()
        if code:
            raise RuntimeError(f"Fresh-data worker {worker} failed: {code}")
        return json.loads((output/"manifest.json").read_text())
    finally:
        output.mkdir(parents=True,exist_ok=True)
        shutil.copy2(logpath,output/"collection.log");volume.commit()


@app.function(image=image,cpu=1,memory=2048,timeout=28800,
              volumes={"/experiments":volume},retries=0,max_containers=1,include_source=False)
def coordinate():
    records=list(collect.map([0,1]))
    volume.reload()
    root=Path("/experiments")
    old=json.loads((root/"iter2000-spatial-data-v3/manifest.json").read_text())
    shards=[]
    for worker,record in enumerate(records):
        if not record["player_checkpoint_unchanged"] or record["checkpoint_sha256"]!=old["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint mismatch")
        shards.extend({**s,"path":f"worker-{worker}/{s['path']}"} for s in record["shards"])
    maps={m for s in shards for m in s["map_ids"]}
    previous={m for split in ("train","validation","test") for s in old[split]["shards"] for m in s["map_ids"]}
    if maps & previous:
        raise ValueError("Fresh audit maps overlap earlier training or evaluation maps")
    count=sum(s["samples"] for s in shards)
    if count!=32768:
        raise ValueError(f"Unexpected snapshot count {count}")
    manifest={"name":DATASET,"checkpoint_sha256":old["checkpoint_sha256"],"config":records[0]["config"],
              "test":{"samples":count,"shards":shards,"unique_maps":len(maps)},
              "map_splits_verified_disjoint":True,"held_out_from":"iter2000-spatial-data-v3",
              "purpose":"Final audit only; never train, tune, or select a reader on this population."}
    (root/DATASET/"manifest.json").write_text(json.dumps(manifest,indent=2)+'\n');volume.commit()
    result={"dataset":DATASET,"snapshots":count,"maps":len(maps),"overlap_with_previous_maps":0}
    print(json.dumps(result),flush=True);return result


@app.local_entrypoint()
def main():
    call=coordinate.spawn();print(json.dumps({"call_id":call.object_id}),flush=True)
