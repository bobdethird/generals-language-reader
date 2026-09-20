"""Longer adapter-only grounding with generated-text factual validation."""
import argparse
import copy
import hashlib
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
from reader.behavior import grounding_caption, BEHAVIOR_PROMPT
from reader.grounding import (expected_facts, summarize_facts, cross_game_shuffle,
                              paired_game_interval, grounding_gate)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode())
        result.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def load_split(path):
    with np.load(path / "samples.npz", allow_pickle=False) as arrays:
        hidden = np.asarray(arrays["activations"], dtype=np.float32)
        observations = np.asarray(arrays["observations"], dtype=np.float32)
    rows = [json.loads(line) for line in (path / "captions.jsonl").read_text().splitlines()]
    if len(hidden) != len(rows) or len(observations) != len(rows):
        raise ValueError("Activation/row count differs")
    return {"hidden": hidden, "rows": rows,
            "captions": [grounding_caption(obs) for obs in observations],
            "facts": [expected_facts(obs) for obs in observations]}


def subset(data, count, seed):
    idx = np.random.default_rng(seed).permutation(len(data["rows"]))[:count]
    return {"hidden": data["hidden"][idx], "indices": idx.tolist(),
            **{key: [data[key][i] for i in idx] for key in ("rows", "captions", "facts")}}


@torch.no_grad()
def evaluate(reader, data, batch_size, conditions=("real", "shuffled", "zero")):
    shuffle = cross_game_shuffle(data["rows"])
    inputs = {"real": data["hidden"], "shuffled": data["hidden"][shuffle],
              "zero": np.zeros_like(data["hidden"])}
    metrics, all_scores, texts = {}, {}, {}
    for condition in conditions:
        generated, total_loss = [], 0.0
        hidden = inputs[condition]
        for start in range(0, len(hidden), batch_size):
            batch = hidden[start:start + batch_size]
            generated.extend(reader.generate(batch, max_new_tokens=80))
            total_loss += float(reader.loss(batch, data["captions"][start:start + len(batch)])) * len(batch)
        metrics[condition], all_scores[condition] = summarize_facts(generated, data["facts"])
        metrics[condition]["caption_nll"] = total_loss / len(hidden)
        texts[condition] = generated
    comparisons = {f"real_vs_{control}": paired_game_interval(all_scores["real"], all_scores[control], data["rows"])
                   for control in conditions if control != "real"}
    return {"metrics": metrics, "comparisons": comparisons}, texts


def save_adapter(path, reader, model_lock, prompt, config, epoch):
    payload = {"adapter": {k: v.detach().cpu() for k, v in reader.adapter.state_dict().items()},
               "activation_dim": config["activation_dim"], "model": model_lock,
               "prompt": prompt, "epoch": epoch, "configuration": config}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--initial-reader", required=True, type=Path)
    parser.add_argument("--player-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["mps", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2000844)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.validation_samples, args.test_samples) <= 0:
        parser.error("Counts must be positive")
    available = torch.cuda.is_available() if args.device == "cuda" else torch.backends.mps.is_available()
    if not available:
        raise RuntimeError(f"Requested GPU backend {args.device} unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    logfile = (args.output / "metrics.jsonl").open("w")

    def log(**data):
        data["seconds"] = round(time.monotonic() - started, 3)
        line = json.dumps(data)
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    manifest = json.loads((args.data / "manifest.json").read_text())
    audit_manifest = json.loads((args.test_data / "manifest.json").read_text())
    checkpoint_hash = digest(args.player_checkpoint)
    if checkpoint_hash != manifest["checkpoint_sha256"] or checkpoint_hash != audit_manifest["checkpoint_sha256"]:
        raise ValueError("Data and frozen player checkpoint do not match")
    if audit_manifest["test"]["map_seed"] in {manifest[k]["map_seed"] for k in ("train", "validation", "test")}:
        raise ValueError("Final audit must have a fresh seed")
    train = load_split(args.data / "train")
    validation = subset(load_split(args.data / "validation"), args.validation_samples, 2000805)
    if {r["game_id"] for r in train["rows"]} & {r["game_id"] for r in validation["rows"]}:
        raise ValueError("Training/validation overlap")
    model_lock = json.loads((ROOT / "reader-model.json").read_text())
    initial = torch.load(args.initial_reader, map_location="cpu", weights_only=True)
    if initial["model"] != model_lock:
        raise ValueError("Language-model revision mismatch")
    prompt = initial.get("prompt", BEHAVIOR_PROMPT)
    path = snapshot_download(model_lock["model"], revision=model_lock["revision"], local_files_only=True)
    reader = LanguageReader(path, train["hidden"].shape[-1], args.device, prompt)
    reader.adapter.load_state_dict(initial["adapter"])
    initial_adapter = copy.deepcopy(reader.adapter.state_dict())
    language_hash_before = model_digest(reader.model)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "model": model_lock, "prompt": prompt, "torch": str(torch.__version__),
              "activation_dim": train["hidden"].shape[-1], "dataset_manifest": manifest,
              "audit_manifest": audit_manifest, "initial_reader_sha256": digest(args.initial_reader),
              "language_sha256_before": language_hash_before,
              "source_sha256": {name: digest(ROOT / name) for name in
                                ["scripts/train_grounding_reader.py", "reader/language.py", "reader/grounding.py", "reader/behavior.py"]},
              "rl_gate": {"each_field_accuracy": .80, "visible_enemy_location_accuracy": .80,
                          "real_minus_shuffled_accuracy": .15, "positive_95_percent_lower_bound": True}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=args.lr)
    adapter_ids = {id(p) for p in reader.adapter.parameters()}
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == adapter_ids
    assert not adapter_ids & {id(p) for p in reader.model.parameters()}
    assert all(not p.requires_grad for p in reader.model.parameters())
    total_steps = math.ceil(len(train["rows"]) / args.batch_size) * args.epochs
    log(stage="start", device=args.device, training_positions=len(train["rows"]),
        epochs=args.epochs, updates=total_steps, presentations=len(train["rows"]) * args.epochs,
        player_frozen=True, language_frozen=True, adapter_parameters=sum(p.numel() for p in reader.adapter.parameters()))
    history = []
    best_key, best_epoch, step = None, None, 0
    for epoch in range(args.epochs + 1):
        if epoch:
            order = rng.permutation(len(train["rows"]))
            for offset in range(0, len(order), args.batch_size):
                selected = order[offset:offset + args.batch_size]
                lr = args.lr * (.1 + .9 * (1 + math.cos(math.pi * step / total_steps)) / 2)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = reader.loss(train["hidden"][selected], [train["captions"][i] for i in selected])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite grounding loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise RuntimeError("Nonfinite adapter gradient")
                optimizer.step()
                step += 1
                if step == 1 or step % 32 == 0:
                    log(stage="train", epoch=epoch, step=step, caption_nll=float(loss.detach()),
                        lr=lr, gradient_norm=float(grad))
        validation_result, examples = evaluate(reader, validation, args.eval_batch_size)
        real = validation_result["metrics"]["real"]
        key = (real["mean_fact_accuracy"], real["all_four_correct"], -real["caption_nll"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            save_adapter(args.output / "best.pt", reader, model_lock, prompt, config, epoch)
        save_adapter(args.output / "latest.pt", reader, model_lock, prompt, config, epoch)
        validation_result.update(epoch=epoch, step=step, selected_best_epoch=best_epoch)
        history.append(validation_result)
        (args.output / f"validation-{epoch}.json").write_text(json.dumps(
            {"evaluation": validation_result, "indices": validation["indices"], "texts": examples}, indent=2) + "\n")
        (args.output / "progress.json").write_text(json.dumps(
            {"epoch": epoch, "step": step, "best_epoch": best_epoch, "total_steps": total_steps}) + "\n")
        log(stage="validation", **validation_result)

    # Load the selected adapter before opening the fresh final-test observations.
    best = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
    reader.adapter.load_state_dict(best["adapter"])
    test = subset(load_split(args.test_data / "test"), args.test_samples, 2000806)
    seen = {r["game_id"] for r in train["rows"] + validation["rows"]}
    if seen & {r["game_id"] for r in test["rows"]}:
        raise ValueError("Final test overlap")
    final_result, final_texts = evaluate(reader, test, args.eval_batch_size)
    reader.adapter.load_state_dict(initial_adapter)
    baseline_result, baseline_texts = evaluate(reader, test, args.eval_batch_size)
    reader.adapter.load_state_dict(best["adapter"])
    language_hash_after = model_digest(reader.model)
    assert language_hash_after == language_hash_before
    assert all(p.grad is None for p in reader.model.parameters())
    assert digest(args.player_checkpoint) == checkpoint_hash
    result = {"configuration": config, "seconds": time.monotonic() - started, "best_epoch": best_epoch,
              "updates": step, "presentations": len(train["rows"]) * args.epochs,
              "validation_history": history, "initial_reader_test": baseline_result, "best_reader_test": final_result,
              "ready_for_rl": grounding_gate(final_result["metrics"]["real"], final_result["comparisons"]["real_vs_shuffled"]),
              "player_checkpoint_unchanged": True, "language_weights_unchanged": True,
              "language_sha256_after": language_hash_after,
              "test_indices": test["indices"],
              "examples": [{"index": index, "game_id": row["game_id"], "visible_facts": caption,
                            "initial": baseline_texts["real"][i], "trained": final_texts["real"][i],
                            "shuffled": final_texts["shuffled"][i], "zero": final_texts["zero"][i]}
                           for i, (index, row, caption) in enumerate(zip(test["indices"], test["rows"], test["captions"]))],
              "scope": "Four observable-fact checks, accepting tied-largest locations and simple region synonyms. "
                       "Unrecognized wording counts as missing. This is not a strategy or causal explanation test; "
                       "accuracy does not validate additional claims outside these four facts. No reader RL was run."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    log(stage="complete", report=str(args.output / "report.json"), best_epoch=best_epoch,
        test=final_result, ready_for_rl=result["ready_for_rl"],
        player_checkpoint_unchanged=True, language_weights_unchanged=True)
    logfile.close()


if __name__ == "__main__":
    main()
