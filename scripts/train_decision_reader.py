"""Train and audit activation-to-language decision descriptions on held-out maps.

Exact action reporting is the first decision-fidelity gate, not a claim that the
decoder has recovered the policy's internal strategic reasoning.
"""
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
from reader.decision_data import load_decision_split
from reader.decision_language import DecisionLanguageReader
from reader.decisions import DECISION_PROMPT, score_decisions, greedy_actions
from reader.grounding import cross_game_shuffle, paired_game_interval
from scripts.train_grounding_reader import digest, model_digest


@torch.no_grad()
def evaluate(reader, data, batch_size, conditions=("real", "shuffled")):
    metrics, texts, correct = {}, {}, {}
    permutation = cross_game_shuffle(data["rows"])
    for condition in conditions:
        source = data["hidden"] if condition == "real" else data["hidden"][permutation]
        if condition == "zero":
            source = np.zeros_like(source)
        elif condition in ("mask_only", "shuffled_h_same_mask"):
            if not reader.legal_mask:
                raise ValueError("Mask controls require a mask-aware reader")
            source = data["hidden"].copy()
            source[..., :448] = 0 if condition == "mask_only" else data["hidden"][permutation, :, :448]
        generated, nll = [], 0.
        for start in range(0, len(source), batch_size):
            batch = source[start:start + batch_size]
            generated.extend(reader.generate(batch, max_new_tokens=90))
            nll += len(batch) * float(reader.loss(batch, data["captions"][start:start + len(batch)]))
        metrics[condition], correct[condition] = score_decisions(generated, data["decision_facts"])
        metrics[condition]["caption_nll"] = nll / len(source)
        texts[condition] = generated
    comparisons = {f"real_vs_{condition}": paired_game_interval(
        correct["real"][:, None], correct[condition][:, None], data["rows"])
        for condition in conditions if condition != "real"}
    return {"metrics": metrics, "comparisons": comparisons}, texts


def sampling_audit(data):
    x = torch.from_numpy(data["logits"])
    cells = x.shape[-1] // 9
    probabilities = x.softmax(-1)
    canonical = torch.cat([probabilities[:, :8*cells], probabilities[:, 8*cells:].sum(-1, keepdim=True)], -1)
    greedy = greedy_actions(data["logits"])
    return {"deployment_greedy_pass_rate": float(np.mean(greedy == 8*cells)),
            "optimal_expected_sample_prediction_accuracy": float(canonical.max(-1).values.mean()),
            "merged_mode_vs_deployment_greedy_disagreement": float(np.mean(canonical.argmax(-1).numpy() != greedy))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--initial-reader", type=Path, required=True)
    p.add_argument("--player-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("spatial", "policy_aux", "policy_mask", "readout"), required=True)
    p.add_argument("--policy-head", type=Path)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=512)
    p.add_argument("--validation-samples", type=int, default=512)
    p.add_argument("--test-samples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=2001034)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA GPU unavailable")
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.eval_every,
           args.validation_samples, args.test_samples) <= 0:
        raise ValueError("Counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    metrics_file = (args.output / "metrics.jsonl").open("w")

    def log(**fields):
        fields["seconds"] = round(time.monotonic() - started, 3)
        line = json.dumps(fields)
        print(line, flush=True)
        metrics_file.write(line + "\n")
        metrics_file.flush()

    manifest = json.loads((args.data / "manifest.json").read_text())
    player_hash = digest(args.player_checkpoint)
    if player_hash != manifest["checkpoint_sha256"] or not manifest["map_splits_verified_disjoint"]:
        raise ValueError("Invalid dataset provenance or split separation")
    log(stage="loading_data", variant=args.variant)
    include_mask = args.variant in ("policy_mask", "readout")
    train = load_decision_split(args.data, "train", include_mask=include_mask)
    validation = load_decision_split(args.data, "validation", count=args.validation_samples, seed=2001015,
                                     include_mask=include_mask)
    train_maps = {r["map_id"] for r in train["rows"]}
    val_maps = {r["map_id"] for r in validation["rows"]}
    if train_maps & val_maps:
        raise ValueError("Map overlap")
    lock = json.loads((ROOT / "reader-model.json").read_text())
    initial = torch.load(args.initial_reader, map_location="cpu", weights_only=True)
    if initial["model"] != lock:
        raise ValueError("Language model revision mismatch")
    path = snapshot_download(lock["model"], revision=lock["revision"], local_files_only=True)
    if args.variant == "readout":
        from reader.readout_language import ReadoutLanguageReader
        if args.policy_head is None:
            raise ValueError("Readout variant requires the original frozen action head")
        head = torch.load(args.policy_head, map_location="cpu", weights_only=True)
        if head["player_sha256"] != player_hash or not head["use_bf16"]:
            raise ValueError("Frozen action-head provenance mismatch")
        reader = ReadoutLanguageReader(path, 448, "cuda", DECISION_PROMPT)
        # This practical branch reuses the completed spatial grounding adapter.
        missing, unexpected = reader.adapter.load_state_dict(initial["adapter"], strict=False)
        if unexpected or any(not k.startswith(("head_", "mask_embedding.", "row_words.", "col_words.",
                                               "direction_words.", "amount_words.", "selected_context.",
                                               "preference.")) for k in missing):
            raise ValueError("Initial spatial adapter is incompatible")
        reader.adapter.head_weight.copy_(head["weight"].to("cuda"))
        reader.adapter.head_bias.copy_(head["bias"].to("cuda"))
        readout_audit = reader.audit_readout(train["hidden"][:4096], train["logits"][:4096])
        log(stage="frozen_head_parity", **readout_audit)
        if readout_audit["greedy_agreement"] < .995:
            raise ValueError("Copied readout does not sufficiently reproduce the original decision")
    else:
        reader = DecisionLanguageReader(path, 448, "cuda", DECISION_PROMPT, policy_aux=args.variant != "spatial",
                                        legal_mask=include_mask)
        reader.adapter.base.load_state_dict(initial["adapter"])
    initial_adapter = copy.deepcopy(reader.adapter.state_dict())
    language_hash = model_digest(reader.model)
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=1e-4, weight_decay=.01)
    assert all(not p.requires_grad for p in reader.model.parameters())
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in reader.adapter.parameters()}
    per_epoch = math.ceil(len(train["rows"]) / args.batch_size)
    total_steps = per_epoch * args.epochs
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "prompt": DECISION_PROMPT, "model": lock, "player_sha256": player_hash,
              "initial_reader_sha256": digest(args.initial_reader), "language_sha256": language_hash,
              "dataset_sha256": digest(args.data / "manifest.json"), "training_snapshots": len(train["rows"]),
              "train_maps": len(train_maps), "validation_maps": len(val_maps),
              "training_presentations": args.epochs * len(train["rows"]), "updates": total_steps,
              "selection": "Move-only exact accuracy, then fully supported descriptions, then NLL",
              "target": "Original raw-logit argmax action used by local greedy opponent",
              "reader_inputs": ("Activations, legal-move mask, and copied frozen original action head" if args.variant == "readout"
                                else "Activations and legal-move mask" if include_mask else "Activations only"),
              "policy_head_sha256": digest(args.policy_head) if args.policy_head else None,
              "source_sha256": {f: digest(ROOT / f) for f in (
                  "scripts/train_decision_reader.py", "reader/decisions.py", "reader/decision_language.py",
                  "reader/decision_data.py", "reader/language.py", "reader/spatial.py", "reader/readout_language.py")}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best_key, best_step = [], None, None

    def save(name, step):
        target = args.output / name
        payload = {"adapter": {k: v.detach().cpu() for k, v in reader.adapter.state_dict().items()},
                   "variant": args.variant, "model": lock, "prompt": DECISION_PROMPT,
                   "activation_dim": 448, "step": step, "configuration": config}
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    def validate(step):
        nonlocal best_key, best_step
        reader.adapter.eval()
        conditions = ("real", "shuffled", "shuffled_h_same_mask") if include_mask else ("real", "shuffled")
        result, texts = evaluate(reader, validation, args.eval_batch_size, conditions=conditions)
        real = result["metrics"]["real"]
        key = (real["move_exact_accuracy"] or 0, real["fully_correct_description"], -real["caption_nll"])
        if best_key is None or key > best_key:
            best_key, best_step = key, step
            save("best.pt", step)
        save("latest.pt", step)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, args.output / "optimizer-latest.pt")
        result.update(step=step, epoch=step/per_epoch, best_step=best_step)
        history.append(result)
        (args.output / f"validation-{step}.json").write_text(json.dumps(
            {"evaluation": result, "indices": validation["indices"], "texts": texts}, indent=2) + "\n")
        log(stage="validation", **result)
        reader.adapter.train()

    log(stage="start", variant=args.variant, updates=total_steps,
        presentations=config["training_presentations"], parameters=sum(p.numel() for p in reader.adapter.parameters()),
        validation_sampling_audit=sampling_audit(validation), player_frozen=True, language_backbone_frozen=True)
    validate(0)
    step = 0
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train["rows"]))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            lr = 1e-4 * (.1 + .9 * (1 + math.cos(math.pi * step/total_steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, stats = reader.decision_loss(train["hidden"][idx], [train["captions"][i] for i in idx],
                train["logits"][idx], spatial=train["spatial"][idx], balance=train["balance"][idx])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite decision loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1)
            if not torch.isfinite(grad):
                raise RuntimeError("Nonfinite decision gradient")
            optimizer.step()
            step += 1
            if step == 1 or step % 32 == 0:
                log(stage="train", step=step, epoch=epoch, loss=float(loss.detach()),
                    lr=lr, gradient_norm=float(grad), **stats)
            if step % args.eval_every == 0 or step == total_steps:
                validate(step)
    selected = torch.load(args.output / "best.pt", map_location="cuda", weights_only=True)
    reader.adapter.load_state_dict(selected["adapter"])
    reader.adapter.eval()
    test = load_decision_split(args.data, "test", count=args.test_samples, seed=2001016, include_mask=include_mask)
    if (train_maps | val_maps) & {r["map_id"] for r in test["rows"]}:
        raise ValueError("Final test overlap")
    log(stage="final_test", best_step=best_step)
    conditions = (("real", "shuffled", "zero", "mask_only", "shuffled_h_same_mask")
                  if include_mask else ("real", "shuffled", "zero"))
    result, texts = evaluate(reader, test, args.eval_batch_size, conditions=conditions)
    reader.adapter.load_state_dict(initial_adapter)
    baseline, _ = evaluate(reader, test, args.eval_batch_size, conditions=("real",))
    reader.adapter.load_state_dict(selected["adapter"])
    assert language_hash == model_digest(reader.model) and digest(args.player_checkpoint) == player_hash
    assert all(p.grad is None for p in reader.model.parameters())
    head_audit = None
    if args.variant == "readout":
        assert torch.equal(reader.adapter.head_weight.cpu(), initial_adapter["head_weight"].cpu())
        assert torch.equal(reader.adapter.head_bias.cpu(), initial_adapter["head_bias"].cpu())
        head_audit = reader.audit_readout(test["hidden"], test["logits"])
    report = {"configuration": config, "best_step": best_step, "seconds": time.monotonic()-started,
              "frozen_head_test": head_audit,
              "validation_history": history, "test": result, "initial_reader_test": baseline,
              "sampling_audit": sampling_audit(test), "test_indices": test["indices"],
              "frozen_player_unchanged": True, "frozen_language_unchanged": True,
              "examples": [{"map_id": row["map_id"], "target": target, "generated": generated,
                            "shuffled": shuffled, "zero": zero}
                           for row, target, generated, shuffled, zero in zip(test["rows"], test["captions"],
                               texts["real"], texts["shuffled"], texts["zero"])],
              "interpretation_scope": "Decision reporting with mechanically verified context. "
                  "Text-only parsing is an explicit-action reconstruction check, not evidence of causal strategic reasoning. "
                  "Further rationale-only and counterfactual audits remain required."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(stage="complete", variant=args.variant, best_step=best_step, test=result,
        frozen_player_unchanged=True, frozen_language_unchanged=True)
    metrics_file.close()


if __name__ == "__main__":
    main()
