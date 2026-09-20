"""Fit matched independent action observers, then audit reader context separately."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from reader.decision_listener import DecisionListener, load_listener_split, context_features
from reader.grounding import cross_game_shuffle, paired_game_interval


def inputs(data, indices, context, device):
    return (torch.as_tensor(data["board"][indices], device=device),
            torch.as_tensor(data["history"][indices], device=device),
            torch.as_tensor(context, dtype=torch.float32, device=device),
            torch.as_tensor(data["legal"][indices], device=device))


@torch.no_grad()
def predict(model, data, context, device, batch_size=128):
    results = []
    for start in range(0, len(data["rows"]), batch_size):
        indices = np.arange(start, min(start+batch_size, len(data["rows"])))
        results.append(model(*inputs(data, indices, context[indices], device)).float().cpu())
    return torch.cat(results)


def summarize(logits, data):
    targets = torch.as_tensor(data["target"])
    correct = (logits.argmax(-1) == targets).numpy()
    moving = data["target"] != 4608
    return {"positions": len(correct), "move_positions": int(moving.sum()),
            "exact_accuracy": float(correct.mean()),
            "move_exact_accuracy": float(correct[moving].mean()) if moving.any() else None,
            "cross_entropy": float(F.cross_entropy(logits, targets))}, correct


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Listener training requires the requested GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    logfile = (args.output / "metrics.jsonl").open("w")

    def log(**fields):
        fields["seconds"] = round(time.monotonic()-started, 3)
        raw = json.dumps(fields)
        print(raw, flush=True)
        logfile.write(raw+'\n')
        logfile.flush()

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["map_splits_verified_disjoint"]:
        raise ValueError("Dataset map splits were not verified")
    log(stage="loading_data")
    training = load_listener_split(args.data, "train")
    validation = load_listener_split(args.data, "validation", count=1024, seed=2001115)
    train_maps = {r["map_id"] for r in training["rows"]}
    if train_maps & {r["map_id"] for r in validation["rows"]}:
        raise ValueError("Train/validation map overlap")
    config = {"training_positions": len(training["rows"]), "train_maps": len(train_maps),
              "dataset_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "player_checkpoint_sha256": manifest["checkpoint_sha256"],
              "epochs": args.epochs, "batch_size": args.batch_size, "width": 384, "layers": 6,
              "seed": 2001124, "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                  for p in ("reader/decision_listener.py", "scripts/train_decision_listener.py")},
              "target": "Deployed raw-logit greedy action; classification, not calibrated policy probabilities",
              "context": "Destination ownership and direction relative to general only; no action sentence or coordinates"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2)+'\n')
    summary = {}
    for kind in ("board_only", "with_context"):
        # Same initialization and batches for the two independent observers.
        torch.manual_seed(config["seed"])
        model = DecisionListener().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        steps_per_epoch = math.ceil(len(training["rows"])/args.batch_size)
        total_steps = steps_per_epoch * args.epochs
        best, best_state, best_step, step = -1., None, None, 0
        validation_context = validation["context"] if kind == "with_context" else np.zeros_like(validation["context"])
        history = []
        for epoch in range(1, args.epochs+1):
            rng = np.random.default_rng(config["seed"]+epoch)
            order = rng.permutation(len(training["rows"]))
            for start in range(0, len(order), args.batch_size):
                selected = order[start:start+args.batch_size]
                features = training["context"][selected].copy()
                drop = rng.random(len(selected)) < .3
                features[drop] = 0
                if kind == "board_only":
                    features[:] = 0
                lr = 3e-4 * (.1 + .9 * (1+math.cos(math.pi*step/total_steps))/2)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(*inputs(training, selected, features, "cuda"))
                    loss = F.cross_entropy(logits.float(), torch.as_tensor(training["target"][selected], device="cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite observer loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                if not torch.isfinite(grad):
                    raise RuntimeError("Nonfinite observer gradient")
                optimizer.step()
                step += 1
                if step == 1 or step % 64 == 0:
                    log(stage="train", observer=kind, epoch=epoch, step=step, loss=float(loss.detach()))
                if step % 256 == 0 or step == total_steps:
                    model.eval()
                    score, _ = summarize(predict(model, validation, validation_context, "cuda"), validation)
                    if score["move_exact_accuracy"] > best:
                        best, best_step = score["move_exact_accuracy"], step
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        torch.save({"model": best_state, "kind": kind, "step": step, "config": config}, args.output/f'{kind}.pt')
                    record = {"step": step, "best_step": best_step, **score}
                    history.append(record)
                    log(stage="validation", observer=kind, **record)
                    model.train()
        summary[kind] = {"best_step": best_step, "validation_history": history}
        del model, optimizer, best_state
        torch.cuda.empty_cache()
    (args.output / "report.json").write_text(json.dumps({"config": config, "observers": summary}, indent=2)+'\n')
    log(stage="complete")
    logfile.close()


def audit(args):
    torch.set_num_threads(8)
    if args.audit_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA audit unavailable")
    data = load_listener_split(args.data, "test", count=1024, seed=2001016)
    board_checkpoint = torch.load(args.output / "board_only.pt", map_location="cpu", weights_only=True)
    context_checkpoint = torch.load(args.output / "with_context.pt", map_location="cpu", weights_only=True)
    models = {}
    for kind, checkpoint in (("board_only", board_checkpoint), ("with_context", context_checkpoint)):
        model = DecisionListener(width=checkpoint["config"]["width"], layers=checkpoint["config"]["layers"])
        model.load_state_dict(checkpoint["model"])
        models[kind] = model.to(args.audit_device).eval().requires_grad_(False)
    board_metrics, board_correct = summarize(predict(models["board_only"], data, np.zeros_like(data["context"]), args.audit_device), data)
    result = {"board_only": board_metrics, "readers": {}, "positions": len(data["rows"]),
              "audit_device": args.audit_device,
              "interpretation_scope": "Independent semantic-context usefulness. Explicit source/direction/split sentences "
                "are removed by construction. This is a behavioral association test, not proof of a strategic motive."}
    permutation = cross_game_shuffle(data["rows"])
    moving = data["target"] != 4608
    for path in args.readers:
        report = json.loads((path / "report.json").read_text())
        if data["indices"] != report["test_indices"]:
            raise ValueError("Reader/observer evaluation positions differ")
        texts = [r["generated"] for r in report["examples"]]
        features = context_features(texts)
        conditions = {"generated_context": features, "oracle_context": data["context"],
                      "no_context": np.zeros_like(features), "shuffled_context": features[permutation]}
        metrics, correctness = {}, {}
        for condition, context in conditions.items():
            metrics[condition], correctness[condition] = summarize(predict(models["with_context"], data, context, args.audit_device), data)
        comparisons = {}
        controls = {"board_only": board_correct, "no_context": correctness["no_context"],
                    "shuffled_context": correctness["shuffled_context"]}
        for control, correct in controls.items():
            comparisons[control] = paired_game_interval(correctness["generated_context"][moving, None],
                correct[moving, None], [r for r, keep in zip(data["rows"], moving) if keep])
        result["readers"][path.name] = {"metrics": metrics, "movement_accuracy_gain": comparisons,
                                      "context_recognition_rate": float((features.sum(-1)>0).mean())}
    target = args.audit_output or args.output / "generated-context-audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--audit-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--audit-output", type=Path)
    p.add_argument("--readers", nargs="+", type=Path)
    args = p.parse_args()
    if args.audit:
        if not args.readers:
            p.error("Audit requires reader report directories")
        audit(args)
    else:
        train(args)
