"""Compare a spatial adapter and the existing adapter on an expanded dataset."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from huggingface_hub import snapshot_download
from reader.language import LanguageReader
from reader.behavior import BEHAVIOR_PROMPT
from reader.grounding import grounding_gate
from reader.spatial_data import load_sharded_split
from scripts.train_grounding_reader import digest, model_digest, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--initial-reader", type=Path, required=True)
    p.add_argument("--player-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", choices=["tokenwise", "spatial"], required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=512)
    p.add_argument("--validation-samples", type=int, default=512)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=5)
    p.add_argument("--aux-weight", type=float, default=.1)
    p.add_argument("--seed", type=int, default=2000944)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Requested NVIDIA GPU unavailable")
    if min(args.epochs, args.batch_size, args.eval_every, args.eval_batch_size) <= 0:
        raise ValueError("Training counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    logfile = (args.output / "metrics.jsonl").open("w")

    def log(**fields):
        fields["seconds"] = round(time.monotonic() - started, 3)
        line = json.dumps(fields)
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    manifest = json.loads((args.data / "manifest.json").read_text())
    checkpoint_hash = digest(args.player_checkpoint)
    if checkpoint_hash != manifest["checkpoint_sha256"] or not manifest["map_splits_verified_disjoint"]:
        raise ValueError("Dataset checkpoint or split verification failed")
    log(stage="loading_data", adapter=args.adapter, samples=manifest["train"]["samples"])
    train = load_sharded_split(args.data, "train")
    validation = load_sharded_split(args.data, "validation", count=args.validation_samples, seed=2000905)
    train_maps = {row["map_id"] for row in train["rows"]}
    val_maps = {row["map_id"] for row in validation["rows"]}
    if train_maps & val_maps:
        raise ValueError("Train/validation map overlap")
    lock = json.loads((ROOT / "reader-model.json").read_text())
    initial = torch.load(args.initial_reader, map_location="cpu", weights_only=True)
    if initial["model"] != lock:
        raise ValueError("Language model revision mismatch")
    prompt = initial.get("prompt", BEHAVIOR_PROMPT)
    path = snapshot_download(lock["model"], revision=lock["revision"], local_files_only=True)
    reader = LanguageReader(path, 448, "cuda", prompt, adapter_type=args.adapter,
                            precision="bfloat16", attention="sdpa")
    if args.adapter == "spatial":
        reader.adapter.base.load_state_dict(initial["adapter"])
    else:
        reader.adapter.load_state_dict(initial["adapter"])
    initial_adapter = copy.deepcopy(reader.adapter.state_dict())
    language_hash = model_digest(reader.model)
    per_epoch = math.ceil(len(train["rows"]) / args.batch_size)
    total_steps = per_epoch * args.epochs
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "model": lock, "precision": "bfloat16", "attention": "sdpa", "torch": str(torch.__version__),
              "training_positions": len(train["rows"]), "train_maps": len(train_maps),
              "validation_maps": len(val_maps), "dataset_manifest_sha256": digest(args.data / "manifest.json"),
              "initial_reader_sha256": digest(args.initial_reader), "player_checkpoint_sha256": checkpoint_hash,
              "language_sha256_before": language_hash, "updates": total_steps,
              "checkpoint_selection": "Fully correct validation descriptions, then individual-fact accuracy, then caption NLL",
              "source_sha256": {name: digest(ROOT / name) for name in
                                ["scripts/train_spatial_reader.py", "reader/language.py", "reader/spatial.py",
                                 "reader/spatial_data.py", "reader/grounding.py", "reader/behavior.py"]}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=args.lr, weight_decay=.01)
    adapter_ids = {id(p) for p in reader.adapter.parameters()}
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == adapter_ids
    assert not adapter_ids & {id(p) for p in reader.model.parameters()}
    assert all(not p.requires_grad for p in reader.model.parameters())
    history, best_key, best_step = [], None, None

    def save(name, step):
        payload = {"adapter": {k: v.detach().cpu() for k, v in reader.adapter.state_dict().items()},
                   "adapter_type": args.adapter, "activation_dim": 448, "model": lock,
                   "precision": "bfloat16", "attention": "sdpa", "prompt": prompt,
                   "step": step, "epoch": step / per_epoch, "configuration": config}
        target = args.output / name
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    def validate(step):
        nonlocal best_key, best_step
        reader.adapter.eval()
        result, texts = evaluate(reader, validation, args.eval_batch_size, conditions=("real", "shuffled"))
        real = result["metrics"]["real"]
        key = (real["all_four_correct"], real["mean_fact_accuracy"], -real["caption_nll"])
        if best_key is None or key > best_key:
            best_key, best_step = key, step
            save("best.pt", step)
        save("latest.pt", step)
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, args.output / "optimizer-latest.pt")
        result.update(step=step, epoch=step / per_epoch, best_step=best_step)
        history.append(result)
        (args.output / f"validation-{step}.json").write_text(json.dumps(
            {"evaluation": result, "indices": validation["indices"], "texts": texts}, indent=2) + "\n")
        (args.output / "progress.json").write_text(json.dumps(
            {"step": step, "total_steps": total_steps, "best_step": best_step, "adapter": args.adapter}) + "\n")
        log(stage="validation", **result)
        reader.adapter.train()

    log(stage="start", adapter=args.adapter, parameters=sum(p.numel() for p in reader.adapter.parameters()),
        updates=total_steps, presentations=args.epochs * len(train["rows"]),
        frozen_player=True, frozen_language_backbone=True)
    validate(0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        # Both model variants see exactly the same shuffled batches.
        order = np.random.default_rng(args.seed + epoch).permutation(len(train["rows"]))
        for offset in range(0, len(order), args.batch_size):
            idx = order[offset:offset + args.batch_size]
            lr = args.lr * (.1 + .9 * (1 + math.cos(math.pi * step / total_steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, stats = reader.training_loss(
                train["hidden"][idx], [train["captions"][i] for i in idx], value_weight=args.value_weight,
                spatial_targets=train["spatial"][idx], balance_targets=train["balance"][idx],
                aux_weight=args.aux_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite reader loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1)
            if not torch.isfinite(gradient):
                raise RuntimeError("Nonfinite reader gradient")
            optimizer.step()
            step += 1
            if step == 1 or step % 32 == 0:
                log(stage="train", epoch=epoch, step=step, **stats, loss=float(loss.detach()),
                    lr=lr, gradient_norm=float(gradient))
            if step % args.eval_every == 0 or step == total_steps:
                validate(step)

    # Final-test examples are opened only after checkpoint selection is complete.
    best = torch.load(args.output / "best.pt", map_location="cuda", weights_only=True)
    reader.adapter.load_state_dict(best["adapter"])
    reader.adapter.eval()
    test = load_sharded_split(args.data, "test", count=args.test_samples, seed=2000906)
    if (train_maps | val_maps) & {row["map_id"] for row in test["rows"]}:
        raise ValueError("Final test map overlap")
    log(stage="final_test", best_step=best_step, samples=len(test["rows"]))
    result, texts = evaluate(reader, test, args.eval_batch_size)
    reader.adapter.load_state_dict(initial_adapter)
    baseline, initial_texts = evaluate(reader, test, args.eval_batch_size, conditions=("real",))
    reader.adapter.load_state_dict(best["adapter"])
    assert model_digest(reader.model) == language_hash
    assert all(p.grad is None for p in reader.model.parameters())
    assert digest(args.player_checkpoint) == checkpoint_hash
    report = {"configuration": config, "adapter": args.adapter, "updates": step,
              "presentations": args.epochs * len(train["rows"]), "best_step": best_step,
              "best_epoch": best_step / per_epoch, "seconds": time.monotonic() - started,
              "validation_history": history, "best_reader_test": result, "initial_reader_test": baseline,
              "ready_for_rl": grounding_gate(result["metrics"]["real"], result["comparisons"]["real_vs_shuffled"]),
              "player_checkpoint_unchanged": True, "language_weights_unchanged": True,
              "test_indices": test["indices"], "test_maps": len({row["map_id"] for row in test["rows"]}),
              "examples": [{"index": index, "game_id": row["game_id"], "visible_facts": caption,
                            "initial": initial_texts["real"][i], "trained": texts["real"][i],
                            "shuffled": texts["shuffled"][i], "zero": texts["zero"][i]}
                           for i, (index, row, caption) in enumerate(zip(test["indices"], test["rows"], test["captions"]))],
              "scope": "Automatically supervised factual descriptions. Spatial heads are training-only; "
                       "the decoder receives continuous activations. No game-policy or language-backbone training; "
                       "no reader RL or action-prediction reward in this experiment."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(stage="complete", adapter=args.adapter, best_step=best_step, test=result,
        ready_for_rl=report["ready_for_rl"], player_checkpoint_unchanged=True, language_weights_unchanged=True)
    logfile.close()


if __name__ == "__main__":
    main()
