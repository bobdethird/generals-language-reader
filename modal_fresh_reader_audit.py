"""Four B200 workers audit the validation-selected reader on untouched snapshots."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT

app=modal.App("generals-fresh-reader-fidelity")
volume=modal.Volume.from_name("generals-reader-experiments",create_if_missing=True)
image=(base_image
       .add_local_file(ROOT/"scripts/audit_fresh_reader.py","/project/scripts/audit_fresh_reader.py")
       .add_local_file(ROOT/"scripts/train_decision_listener.py","/project/scripts/train_decision_listener.py")
       .add_local_file(ROOT/"modal_fresh_reader_audit.py","/project/modal_fresh_reader_audit.py"))
DATASET="iter2000-fresh-reader-audit-v1"
OUTPUT="iter2000-fresh-reader-results-v1"


@app.function(image=image,gpu="B200",cpu=8,memory=32768,timeout=3600,
              volumes={"/experiments":volume},retries=0,max_containers=4,
              single_use_containers=True,scaledown_window=2,include_source=False)
def evaluate(worker:int,reader_name:str):
    import re
    import subprocess
    import sys
    volume.reload()
    if not 0<=worker<4 or not re.fullmatch(r"[a-zA-Z0-9_-]+",reader_name):
        raise ValueError("Invalid audit assignment")
    root=Path("/experiments");output=root/OUTPUT/f"worker-{worker}.json"
    if output.exists():
        raise FileExistsError(output)
    hardware=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],text=True).strip()
    if "B200" not in hardware:
        raise RuntimeError("Expected B200")
    command=[sys.executable,"-u","/project/scripts/audit_fresh_reader.py","--data",str(root/DATASET),
             "--reader",str(root/reader_name),"--observers",str(root/"iter2000-decision-listener-v1"),
             "--output",str(output),"--worker",str(worker)]
    print(json.dumps({"hardware":hardware,"command":command}),flush=True)
    try:
        subprocess.run(command,cwd="/project",check=True)
        r=json.loads(output.read_text())
        return {"worker":worker,"positions":r["positions"],"metrics":r["metrics"]}
    finally:
        volume.commit()


@app.function(image=image,cpu=2,memory=4096,timeout=28800,
              volumes={"/experiments":volume},retries=0,max_containers=1,include_source=False)
def coordinate():
    import numpy as np
    from reader.decisions import parse_decision,score_decisions
    from reader.grounding import paired_game_interval
    modal.FunctionCall.from_id("fc-01M2YTAV86BW1C2X49E7R7N381").get()
    selection=modal.FunctionCall.from_id("fc-01M2YTC74RPSJX4PJND78RMBCT").get()
    reader=selection["selected_reader"]
    volume.reload();root=Path("/experiments");output=root/OUTPUT
    # A CPU coordinator can be interrupted after committing successful workers.
    # Resume the same audit, checking identities, instead of discarding GPU work.
    output.mkdir(exist_ok=True)
    # Registered before the fresh population is evaluated. This is a transfer
    # pilot gate, not a declaration that causal strategic motives are recovered.
    criteria={"exact_move":.95,"fully_supported_description":.85,"half_move_exact":.90,
              "minimum_half_positions":100,"context_gain_ci_lower_above":0.0}
    plan={"selected_reader":reader,"selection":"validation only","transfer_pilot_gate":criteria}
    if (output/"plan.json").exists() and json.loads((output/"plan.json").read_text())!=plan:
        raise ValueError("Existing fresh-audit plan does not match")
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+'\n');volume.commit()
    missing=[(worker,reader) for worker in range(4) if not (output/f"worker-{worker}.json").exists()]
    if missing:
        list(evaluate.starmap(missing))
    volume.reload()
    pieces=[json.loads((output/f"worker-{worker}.json").read_text()) for worker in range(4)]
    import hashlib
    reader_hash=hashlib.sha256((root/reader/"best.pt").read_bytes()).hexdigest()
    dataset_hash=hashlib.sha256((root/DATASET/"manifest.json").read_bytes()).hexdigest()
    for worker,piece in enumerate(pieces):
        if (piece["worker"]!=worker or piece["reader"]!=reader
                or piece["reader_checkpoint_sha256"]!=reader_hash
                or piece["dataset_manifest_sha256"]!=dataset_hash
                or piece["positions"]!=len(piece["rows"])):
            raise ValueError("Persisted audit worker identity or size mismatch")
    manifest=json.loads((root/DATASET/"manifest.json").read_text())
    paths=[path for piece in pieces for path in piece["shards"]]
    expected={s["path"] for s in manifest["test"]["shards"]}
    if set(paths)!=expected or len(paths)!=len(expected):
        raise ValueError("Fresh shard partitions overlap or omit data")
    if len({p["reader_checkpoint_sha256"] for p in pieces})!=1:
        raise ValueError("Workers evaluated different reader checkpoints")
    rows=[row for piece in pieces for row in piece["rows"]]
    if len(rows)!=32768 or sum(p["positions"] for p in pieces)!=len(rows):
        raise ValueError("Fresh audit size mismatch")
    groups=[{"game_id":r["map_id"]} for r in rows]
    texts=[r["generated"] for r in rows];targets=[parse_decision(r["target"]) for r in rows]
    if any(t is None for t in targets):
        raise ValueError("Invalid automatic reference description")
    metrics,correct=score_decisions(texts,targets)
    moving=np.array([r["moving"] for r in rows]);half=np.array([r["half"] for r in rows])
    def interval(values,indices):
        selected=np.flatnonzero(indices)
        return paired_game_interval(np.asarray(values)[selected,None],np.zeros((len(selected),1)),
                                    [groups[i] for i in selected])["game_bootstrap_95_percent_interval"]
    breakdown={}
    for label,index in (("wait",~moving),("full_moves",moving & ~half),("half_moves",half)):
        breakdown[label]={"positions":int(index.sum()),"exact_accuracy":float(correct[index].mean()) if index.any() else None,
                          "accuracy_95_percent_interval":interval(correct,index) if index.any() else None}
    observer={}
    flags={name:np.array([r["observer_correct"][name] for r in rows]) for name in rows[0]["observer_correct"]}
    for name,values in flags.items():
        observer[name]={"exact_accuracy":float(values.mean()),"movement_accuracy":float(values[moving].mean())}
    gains={name:paired_game_interval(flags["generated_context"][moving,None],flags[name][moving,None],
            [g for g,keep in zip(groups,moving) if keep]) for name in ("board_only","no_context","shuffled_context")}
    checks={"exact_move":metrics["exact_action_accuracy"]>=criteria["exact_move"],
            "fully_supported_description":metrics["fully_correct_description"]>=criteria["fully_supported_description"],
            "half_move_exact":bool(half.sum()>=criteria["minimum_half_positions"] and correct[half].mean()>=criteria["half_move_exact"]),
            "context_gain":gains["board_only"]["game_bootstrap_95_percent_interval"][0]>0}
    report={"selected_reader":reader,"positions":len(rows),"maps":len({r["map_id"] for r in rows}),
            "metrics":metrics,"exact_accuracy_95_percent_interval":interval(correct,np.ones(len(rows),bool)),
            "by_action_type":breakdown,"independent_observer":observer,"context_movement_gains":gains,
            "copied_head_agreement":float(np.mean([r["head_correct"] for r in rows])) if rows[0]["head_correct"] is not None else None,
            "transfer_pilot_gate":{"criteria":criteria,"checks":checks,"passed":all(checks.values())},
            "reader_checkpoint_sha256":pieces[0]["reader_checkpoint_sha256"],
            "dataset_manifest_sha256":pieces[0]["dataset_manifest_sha256"],
            "scope":"Fresh-data decision reporting and descriptive-context usefulness. A passing transfer gate supports "
                    "a new checkpoint pilot; it does not prove causal strategic reasoning. The frozen observers have finite capacity and training."}
    (output/"report.json").write_text(json.dumps(report,indent=2)+'\n');volume.commit()
    print(json.dumps(report),flush=True);return report


@app.local_entrypoint()
def main():
    call=coordinate.spawn();print(json.dumps({"call_id":call.object_id}),flush=True)
