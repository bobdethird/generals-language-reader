"""Collect measured checkpoint-3800 influences and train the post-hoc decoder."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("generals-3800-why-language")
volume = modal.Volume.from_name("generals-reader-experiments")
DATA = "iter3800-spatial-data-v1"
EVIDENCE = "iter3800-counterfactual-data-v2"
READER = "iter3800-why-reader-v2"
PLAYER = "iter3800-reader-transfer-v1/L_7d_gae90_ema_3800.eqx"
if modal.is_local():
    from modal_policy import image as collection_base
    from modal_spatial_training import image as language_base
    collection_image = collection_base.add_local_file(ROOT/"modal_why_training.py", "/project/modal_why_training.py")
    language_image = (language_base
        .add_local_file(ROOT/"modal_why_training.py", "/project/modal_why_training.py")
        .add_local_file(ROOT/"scripts/train_why_reader.py", "/project/scripts/train_why_reader.py"))
else:
    collection_image = language_image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=collection_image, gpu="B200", cpu=8, memory=32768, timeout=10800,
              volumes={"/experiments":volume}, retries=0, max_containers=4,
              single_use_containers=True, scaledown_window=2, include_source=False)
def collect(worker):
    from scripts.collect_counterfactuals import extract
    import hashlib
    volume.reload(); root = Path("/experiments")
    manifest = json.loads((root/DATA/"manifest.json").read_text())
    player_hash = hashlib.sha256((root/PLAYER).read_bytes()).hexdigest()
    if player_hash != manifest["checkpoint_sha256"]: raise ValueError("Player identity mismatch")
    tasks = [(split, shard, count) for split,count in (("train",512),("validation",1024),("test",2048))
             for shard in manifest[split]["shards"]]
    records=[]
    try:
        for index in range(worker,len(tasks),4):
            split,shard,count=tasks[index]
            name=f"{split}-{index:03d}"
            output=root/EVIDENCE/name
            if (output/"report.json").exists():
                report=json.loads((output/"report.json").read_text())
                if report["player_sha256"]!=player_hash or report["positions"]!=count:
                    raise ValueError("Persisted evidence identity/count mismatch")
            else:
                report=extract(root/PLAYER,root/DATA/shard["path"],output,count,380100+index)
            records.append({"split":split,"path":name,"report":report})
            volume.commit()
        return records
    finally:
        volume.commit()


@app.function(image=language_image, gpu="B200", cpu=8, memory=65536, timeout=10800,
              volumes={"/experiments":volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def train(architecture="tokenwise",output_name=""):
    import subprocess
    import sys
    import re
    if architecture not in ("tokenwise","structured","pointer"):raise ValueError("Invalid architecture")
    name=output_name or READER
    if not re.fullmatch(r"[A-Za-z0-9_-]+",name):raise ValueError("Invalid output name")
    volume.reload(); root=Path("/experiments"); output=root/name
    if architecture in ("structured","pointer"):
        plan={"candidates":[READER,name],"data":EVIDENCE,"selection":"Best validation exact rationale accuracy only",
              "reason":"Initial tokenwise reader confuses the strongest influence; compare a learned evidence-comparison layer",
              "same_epochs":8,"same_seed":380031,"same_splits":True,
              "architecture_change":"Two attention layers plus training-only auxiliary targets; inference gets numerical evidence, never reference labels"}
        plan["architecture"]=architecture
        if architecture=="pointer":
            plan["candidates"]=[READER,"iter3800-why-reader-v3",name]
            plan["architecture_change"]="Shared learned per-probe scorer, with eligibility from measured tests; no reference labels at inference"
        (root/f"iter3800-why-{architecture}-plan.json").write_text(json.dumps(plan,indent=2)+'\n');volume.commit()
    if (output/"report.json").exists(): return json.loads((output/"report.json").read_text())
    command=[sys.executable,"-u","/project/scripts/train_why_reader.py","--data",str(root/EVIDENCE),
             "--output",str(output),"--epochs","8","--architecture",architecture]
    try:
        p=subprocess.Popen(command,cwd="/project",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            print(line,end="",flush=True)
            if line.startswith('{"stage": "validation"'): volume.commit()
        if p.wait(): raise RuntimeError(f"Why training exited {p.returncode}")
        return json.loads((output/"report.json").read_text())
    finally:
        volume.commit()


@app.function(image=collection_image,cpu=1,memory=4096,timeout=28800,
              volumes={"/experiments":volume},retries=0,max_containers=1,include_source=False)
def coordinate(transfer_call, pilot_call):
    import time
    pilot=modal.FunctionCall.from_id(pilot_call).get()
    if pilot["positions"]!=256 or pilot["nontrivial_explanations"]<25:
        raise ValueError("Counterfactual pilot has insufficient usable influence evidence")
    root=Path("/experiments")
    dependency=modal.FunctionCall.from_id(transfer_call)
    while True:
        volume.reload()
        if (root/DATA/"manifest.json").exists(): break
        try:
            result=dependency.get(timeout=0)
            raise RuntimeError(f"Transfer ended before producing data: {result.get('status')}")
        except TimeoutError:
            time.sleep(20)
    source=json.loads((root/DATA/"manifest.json").read_text())
    if not source["map_splits_verified_disjoint"]: raise ValueError("Unverified map split")
    output=root/EVIDENCE;output.mkdir(exist_ok=True)
    plan={"source":DATA,"player_sha256":source["checkpoint_sha256"],"train":16384,
          "validation":2048,"test":4096,"selection":"Validation only; final test is opened once after weight selection",
          "success_criteria":{"exact_rationale_accuracy":.95,"nontrivial_accuracy":.95,
                              "shuffled_evidence_gain_ci_lower":0.0}}
    if (output/"plan.json").exists() and json.loads((output/"plan.json").read_text())!=plan:
        raise ValueError("Existing counterfactual plan differs")
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+'\n');volume.commit()
    groups=list(collect.map(range(4)))
    volume.reload()
    records=[r for group in groups for r in group]
    manifest={"player_sha256":source["checkpoint_sha256"],"source_dataset":DATA,"plan":plan}
    maps=[]
    for split in ("train","validation","test"):
        selected=[r for r in records if r["split"]==split]
        if sum(r["report"]["positions"] for r in selected)!=plan[split]: raise ValueError("Evidence count mismatch")
        manifest[split]=[r["path"] for r in selected]
        maps.append({m for r in selected for m in r["report"]["map_ids"]})
    if any(a & b for i,a in enumerate(maps) for b in maps[i+1:]): raise ValueError("Evidence map splits overlap")
    manifest["map_splits_verified_disjoint"]=True
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+'\n');volume.commit()
    report=train.remote()
    return {"status":"complete","reader":READER,"best_step":report["best_step"],"test":report["test"]}


@app.local_entrypoint()
def main(transfer_call: str="", pilot_call: str="fc-01M2YW17203TZ9C0XMPTDB5T2D",architecture: str="tokenwise"):
    if architecture in ("structured","pointer"):
        name="iter3800-why-reader-v3" if architecture=="structured" else "iter3800-why-reader-v4"
        call=train.spawn(architecture,name)
    elif architecture=="tokenwise" and transfer_call:
        call=coordinate.spawn(transfer_call,pilot_call)
    else:raise ValueError("Specify a transfer call or the structured comparison")
    print(json.dumps({"call_id":call.object_id,"transfer_call":transfer_call,"pilot_call":pilot_call}),flush=True)
