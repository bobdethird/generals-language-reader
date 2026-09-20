"""Train the post-hoc evidence decoder; select on validation, audit unseen maps."""
import argparse
import hashlib
import json
import math
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
from huggingface_hub import snapshot_download
from reader.why_language import WhyLanguageReader, pack_why_inputs, WHY_PROMPT
from reader.counterfactuals import parse_rationale, PROBES, SCOPE
from reader.grounding import paired_game_interval, cross_game_shuffle
from scripts.train_grounding_reader import model_digest


def load_split(root, manifest, split):
    hidden, evidence, rows = [], [], []
    for shard in manifest[split]:
        path = root/shard
        report = json.loads((path/"report.json").read_text())
        if report["player_sha256"] != manifest["player_sha256"]:
            raise ValueError("Mixed player checkpoints")
        with np.load(path/"evidence.npz", allow_pickle=False) as z:
            hidden.append(z["activations"]); evidence.append(z["evidence"])
        rows.extend(json.loads(line) for line in (path/"examples.jsonl").read_text().splitlines())
    return pack_why_inputs(np.concatenate(hidden), np.concatenate(evidence)), rows


def scores(texts, rows):
    parsed = [parse_rationale(t) for t in texts]
    target = [r["rationale"] for r in rows]
    exact = np.array([p == t for p, t in zip(parsed, target)])
    active = np.array([t["probe"] >= 0 for t in target])
    def conditional(field):
        return float(np.mean([p is not None and p[field] == t[field]
                              for p, t, keep in zip(parsed, target, active) if keep])) if active.any() else None
    by_probe={}
    for k in range(-1,len(PROBES)):
        selected=np.array([t["probe"]==k for t in target])
        by_probe["inconclusive" if k<0 else PROBES[k]]={"positions":int(selected.sum()),
            "exact_accuracy":float(exact[selected].mean()) if selected.any() else None}
    return {"positions": len(rows), "parse_rate": float(np.mean([p is not None for p in parsed])),
            "exact_rationale_accuracy": float(exact.mean()),
            "nontrivial_positions": int(active.sum()), "nontrivial_coverage": float(active.mean()),
            "nontrivial_accuracy": float(exact[active].mean()) if active.any() else None,
            "probe_accuracy": conditional("probe"), "effect_sign_accuracy": conditional("effect"),
            "switch_accuracy": conditional("switches"), "by_probe":by_probe}, exact


@torch.no_grad()
def evaluate(reader, data, rows, batch_size=64, controls=False):
    generated, metrics, correct = {}, {}, {}
    conditions = ["real", "shuffled_evidence", "no_evidence"] if controls else ["real"]
    shuffle = cross_game_shuffle([{**r, "game_id": r["map_id"]} for r in rows]) if controls else None
    for name in conditions:
        texts = []
        for start in range(0, len(data), batch_size):
            x = data[start:start+batch_size].copy()
            if name == "shuffled_evidence": x[:, 67:] = data[shuffle[start:start+len(x)], 67:]
            if name == "no_evidence": x[:, 67:] = 0
            texts.extend(reader.generate(x, max_new_tokens=80))
        metrics[name], correct[name] = scores(texts, rows)
        generated[name] = texts
    groups = [{"game_id": r["map_id"]} for r in rows]
    ci = paired_game_interval(correct["real"][:, None], np.zeros((len(rows), 1)), groups)
    gains = {k: paired_game_interval(correct["real"][:, None], correct[k][:, None], groups)
             for k in conditions if k != "real"}
    return {"metrics": metrics, "exact_accuracy_map_bootstrap": ci, "control_gains": gains}, generated


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=256)
    p.add_argument("--architecture",choices=("tokenwise","structured","pointer"),default="tokenwise")
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if min(args.epochs, args.batch_size, args.eval_every) < 1: raise ValueError("Invalid training counts")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(380031); torch.set_num_threads(8)
    manifest = json.loads((args.data/"manifest.json").read_text())
    train, train_rows = load_split(args.data, manifest, "train")
    val, val_rows = load_split(args.data, manifest, "validation")
    map_sets = [{r["map_id"] for r in rows} for rows in (train_rows, val_rows)]
    if map_sets[0] & map_sets[1]: raise ValueError("Training/validation map overlap")
    lock = json.loads((ROOT/"reader-model.json").read_text())
    path = snapshot_download(lock["model"], revision=lock["revision"], local_files_only=True)
    reader = WhyLanguageReader(path, "cuda",args.architecture)
    frozen_hash = model_digest(reader.model)
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=3e-4, weight_decay=.01)
    total_steps = math.ceil(len(train)/args.batch_size)*args.epochs
    config = {"model": lock,"architecture":args.architecture,"player_sha256": manifest["player_sha256"], "epochs": args.epochs,
              "train_positions": len(train), "validation_positions": len(val), "total_updates": total_steps,
              "prompt": WHY_PROMPT, "language_sha256": frozen_hash,
              "manifest_sha256": hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                ["reader/counterfactuals.py", "reader/why_language.py", "scripts/train_why_reader.py"]},
              "input_contract": "Detached hidden activations and measured numerical intervention effects. No rationale words or labels.",
              "selection": "Validation exact-rationale accuracy; no selection on final test", "scope": SCOPE}
    (args.output/"config.json").write_text(json.dumps(config, indent=2)+'\n')
    started = time.monotonic(); log_file = (args.output/"metrics.jsonl").open("w")
    def log(**fields):
        line = json.dumps({**fields, "seconds": round(time.monotonic()-started, 2)})
        print(line, flush=True); log_file.write(line+'\n'); log_file.flush()
    best, best_step, step = -1., 0, 0
    def save(name):
        tmp = args.output/(name+'.tmp')
        torch.save({"variant": "why","architecture":args.architecture,"model": lock, "adapter": {k:v.cpu() for k,v in reader.adapter.state_dict().items()},
                    "step": step, "configuration": config}, tmp)
        tmp.replace(args.output/name)
    def validate():
        nonlocal best, best_step
        reader.adapter.eval()
        result, texts = evaluate(reader, val, val_rows)
        accuracy = result["metrics"]["real"]["exact_rationale_accuracy"]
        if accuracy > best:
            best, best_step = accuracy, step; save("best.pt")
        save("latest.pt")
        (args.output/f"validation-{step}.json").write_text(json.dumps({"step": step, **result, "texts": texts}, indent=2)+'\n')
        log(stage="validation", step=step, best_step=best_step, **result)
        reader.adapter.train()
    log(stage="start", updates=total_steps, train_positions=len(train), scope=SCOPE)
    probe_targets=np.array([r["rationale"]["probe"]+1 for r in train_rows])
    effect_targets=np.array([{"none":0,"weakens":1,"strengthens":2}[r["rationale"]["effect"]] for r in train_rows])
    switch_targets=np.array([int(r["rationale"]["switches"]) for r in train_rows])
    counts=np.bincount(probe_targets,minlength=11).clip(1)
    probe_weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device="cuda")
    probe_weights/=probe_weights.mean()
    validate()
    for epoch in range(args.epochs):
        order = np.random.default_rng(380031+epoch).permutation(len(train))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start+args.batch_size]
            for group in optimizer.param_groups:
                group["lr"] = 3e-4*(.1+.9*(1+math.cos(math.pi*step/total_steps))/2)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = reader.training_loss(train[idx], [train_rows[i]["why_target"] for i in idx], value_weight=1.)
            if args.architecture in ("structured","pointer"):
                from torch.nn import functional as F
                _,aux=reader.adapter.forward_with_aux(torch.as_tensor(train[idx],device="cuda").detach())
                loss=loss+.3*F.cross_entropy(aux["probe"],torch.as_tensor(probe_targets[idx],device="cuda"),weight=probe_weights)
                loss=loss+.1*F.cross_entropy(aux["effect"],torch.as_tensor(effect_targets[idx],device="cuda"))
                loss=loss+.1*F.cross_entropy(aux["switch"],torch.as_tensor(switch_targets[idx],device="cuda"))
            if not torch.isfinite(loss): raise ValueError("Nonfinite why loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1.)
            optimizer.step(); step += 1
            if step == 1 or step % 32 == 0: log(stage="train", step=step, loss=float(loss.detach()))
            if step % args.eval_every == 0 or step == total_steps: validate()
    selected = torch.load(args.output/"best.pt", weights_only=True, map_location="cuda")
    reader.adapter.load_state_dict(selected["adapter"]); reader.adapter.eval()
    test, test_rows = load_split(args.data, manifest, "test")
    if (map_sets[0] | map_sets[1]) & {r["map_id"] for r in test_rows}: raise ValueError("Final test map overlap")
    result, texts = evaluate(reader, test, test_rows, controls=True)
    if model_digest(reader.model) != frozen_hash: raise ValueError("Language backbone changed")
    if any(p.grad is not None for p in reader.model.parameters()): raise ValueError("Backbone received gradients")
    report = {"configuration": config, "best_step": best_step, "test": result,
              "language_unchanged": True, "examples": [{"map_id": r["map_id"], "target": r["why_target"],
                  "generated": text, "shuffled_evidence": shuffled} for r,text,shuffled in
                  zip(test_rows,texts["real"],texts["shuffled_evidence"])], "scope": SCOPE}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+'\n')
    log(stage="complete", best_step=best_step, test=result)


if __name__ == "__main__": main()
