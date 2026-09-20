"""Evaluate one validation-selected reader on a disjoint shard partition of fresh data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT/".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from reader.decision_data import load_decision_split
from reader.decision_factory import restore_reader
from reader.decisions import score_decisions, greedy_actions
from reader.decision_listener import DecisionListener, load_listener_split, context_features
from reader.grounding import cross_game_shuffle
from scripts.train_decision_listener import predict, summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--reader",type=Path,required=True)
    p.add_argument("--observers",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",type=int,required=True)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if not torch.cuda.is_available() or not 0<=args.worker<args.workers:
        raise ValueError("Invalid CUDA audit request")
    torch.set_num_threads(8);torch.set_float32_matmul_precision("high")
    started=time.monotonic()
    manifest=json.loads((args.data/"manifest.json").read_text())
    shards=manifest["test"]["shards"][args.worker::args.workers]
    if not shards or not manifest["map_splits_verified_disjoint"]:
        raise ValueError("Fresh audit shard assignment or provenance is invalid")
    # A read-only view selects disjoint whole shards; source dataset stays intact.
    view=Path(f"/tmp/fresh-reader-view-{args.worker}");view.mkdir(exist_ok=False)
    subset={**manifest,"test":{**manifest["test"],"shards":[{**s,"path":str(args.data/s['path'])} for s in shards]}}
    subset["test"]["samples"]=sum(s["samples"] for s in shards)
    (view/"manifest.json").write_text(json.dumps(subset))
    checkpoint=torch.load(args.reader/"best.pt",map_location="cpu",weights_only=True)
    if checkpoint["configuration"]["player_sha256"]!=manifest["checkpoint_sha256"]:
        raise ValueError("Reader was trained on a different player")
    reader=restore_reader(checkpoint,"cuda")
    data=load_decision_split(view,"test",include_mask=reader.legal_mask)
    if len(data["rows"])!=subset["test"]["samples"]:
        raise ValueError("Audit shard row count mismatch")
    texts=[]
    for start in range(0,len(data["rows"]),64):
        texts.extend(reader.generate(data["hidden"][start:start+64],max_new_tokens=90))
        if start%1024==0:
            print(json.dumps({"stage":"generate","worker":args.worker,"completed":len(texts),
                              "total":len(data["rows"]),"seconds":time.monotonic()-started}),flush=True)
    metrics,correct=score_decisions(texts,data["decision_facts"])
    fully=np.array([score_decisions([text],[target])[0]["fully_correct_description"]==1
                    for text,target in zip(texts,data["decision_facts"])])
    head_correct=None
    if checkpoint["variant"]=="readout":
        head_correct=[]
        for start in range(0,len(texts),128):
            logits=reader.adapter.read_logits(torch.as_tensor(data["hidden"][start:start+128],device="cuda"))
            predicted=logits.argmax(-1).clamp_max(4608).cpu().numpy()
            head_correct.extend((predicted==greedy_actions(data["logits"][start:start+128])).tolist())
    del reader
    torch.cuda.empty_cache()
    observer_data=load_listener_split(view,"test")
    if observer_data["captions"]!=data["captions"] or observer_data["indices"]!=data["indices"]:
        raise ValueError("Observer and language audit populations disagree")
    features=context_features(texts)
    permutation=cross_game_shuffle(observer_data["rows"])
    observer_metrics,observer_correct={},{}
    for kind in ("board_only","with_context"):
        saved=torch.load(args.observers/f"{kind}.pt",map_location="cpu",weights_only=True)
        model=DecisionListener(width=saved["config"]["width"],layers=saved["config"]["layers"]).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved["model"])
        conditions=({"board_only":np.zeros_like(features)} if kind=="board_only" else {
            "generated_context":features,"oracle_context":observer_data["context"],
            "no_context":np.zeros_like(features),"shuffled_context":features[permutation]})
        for condition,context in conditions.items():
            observer_metrics[condition],observer_correct[condition]=summarize(predict(model,observer_data,context,"cuda"),observer_data)
        del model
        torch.cuda.empty_cache()
    rows=[]
    for i,(row,target,text,facts) in enumerate(zip(data["rows"],data["captions"],texts,data["decision_facts"])):
        rows.append({"map_id":row["map_id"],"target":target,"generated":text,"moving":not facts["pass"],
                     "half":not facts["pass"] and facts["half"],"exact":bool(correct[i]),"complete":bool(fully[i]),
                     "head_correct":head_correct[i] if head_correct is not None else None,
                     "observer_correct":{k:bool(v[i]) for k,v in observer_correct.items()}})
    report={"worker":args.worker,"positions":len(rows),"shards":[s["path"] for s in shards],
            "reader":args.reader.name,"reader_checkpoint_sha256":hashlib.sha256((args.reader/"best.pt").read_bytes()).hexdigest(),
            "dataset_manifest_sha256":hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest(),
            "metrics":metrics,"observer_metrics":observer_metrics,"rows":rows,"seconds":time.monotonic()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({"stage":"complete","worker":args.worker,"metrics":metrics,"seconds":report["seconds"]}),flush=True)


if __name__=="__main__":
    main()
