"""Bounded, local text-to-behavior reconstruction experiment.

No manually labeled data or remote APIs. The player and language backbone stay
frozen. We warm-start English from visible facts, fit separate move predictors,
freeze them, and train the activation adapter with sampled-text policy gradients.
An independently initialized auditor never supplies the reader's reward. A new
test-map seed is opened only for final evaluation; it never selects checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
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
from reader.behavior import (BEHAVIOR_PROMPT, BehaviorReader, MovePredictor,
                             grounding_caption, normalize_inputs, canonical_targets,
                             canonical_actions, legal_actions, behavior_scores,
                             distillation_loss, group_advantages)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path):
    with np.load(path / "samples.npz", allow_pickle=False) as f:
        data = {key: np.array(value) for key, value in f.items()}
    rows = [json.loads(s) for s in (path / "captions.jsonl").read_text().splitlines()]
    if len(rows) != len(data["activations"]):
        raise ValueError("Caption/activation count mismatch")
    obs, temporal = normalize_inputs(data["observations"], data["temporal"])
    data.update(board=torch.from_numpy(obs), history=torch.from_numpy(temporal),
                target=canonical_targets(data["logits"]), legal=legal_actions(data["masks"]),
                sampled=torch.from_numpy(canonical_actions(data["actions"], obs.shape[-1])),
                captions=[grounding_caption(o) for o in data["observations"]], rows=rows)
    return data


def inputs(data, idx, features, device):
    return [data["board"][idx].to(device), data["history"][idx].to(device),
            features.to(device), data["legal"][idx].to(device)]


@torch.no_grad()
def predict(model, data, idx, features, device, batch_size=64):
    outputs = []
    for offset in range(0, len(idx), batch_size):
        selected = idx[offset:offset + batch_size]
        outputs.append(model(*inputs(data, selected, features[offset:offset + batch_size], device)).cpu())
    return torch.cat(outputs)


def summarize(logits, data, idx):
    scores = behavior_scores(logits, data["target"][idx], data["sampled"][idx])
    result = {}
    for key, values in scores.items():
        if key == "has_moves":
            continue
        use = scores["has_moves"] if key.startswith("move_") else torch.ones_like(scores["has_moves"])
        result[key] = float(values[use].mean()) if use.any() else None
    result["positions"] = len(idx)
    result["positions_with_legal_moves"] = int(scores["has_moves"].sum())
    return result, scores


@torch.no_grad()
def generate_descriptions(wrapper, hidden, batch_size, max_tokens, log, label):
    texts = []
    for start in range(0, len(hidden), batch_size):
        _, generated = wrapper.sample(hidden[start:start + batch_size], max_new_tokens=max_tokens, sample=False)
        texts.extend(generated)
        if start == 0 or (start // batch_size + 1) % 8 == 0:
            log(stage=label, generated=len(texts), total=len(hidden))
    return texts


def evaluate_descriptions(wrapper, models, data, idx, texts, device):
    features = wrapper.text_features(texts)
    # Draw descriptions from other games, never nearby positions in this game.
    games = np.array([data["rows"][int(i)]["game_id"] for i in idx])
    permutation = np.arange(len(idx))
    rng = np.random.default_rng(718)
    for i in range(len(idx)):
        choices = np.flatnonzero(games != games[i])
        if not len(choices):
            raise ValueError("Shuffled-text controls require multiple held-out games")
        permutation[i] = rng.choice(choices)
    result, raw = {}, {}
    for name, model in models.items():
        conditions = {"no_text": torch.zeros_like(features)} if name == "board_only" else {
            "real_text": features, "no_text": torch.zeros_like(features),
            "shuffled_text": features[permutation]}
        result[name], raw[name] = {}, {}
        for condition, f in conditions.items():
            logits = predict(model, data, idx, f, device)
            result[name][condition], raw[name][condition] = summarize(logits, data, idx)
    return result, raw, features


def paired_game_interval(before, after, rows, idx, seed=442):
    """Positive is improvement; resample whole games, not correlated snapshots."""
    delta = (before["move_kl"] - after["move_kl"]).numpy()
    usable = before["has_moves"].numpy()
    groups = {}
    for j, i in enumerate(idx):
        if usable[j]:
            groups.setdefault(rows[int(i)]["game_id"], []).append(float(delta[j]))
    values = [np.array(x) for x in groups.values()]
    rng = np.random.default_rng(seed)
    samples = [np.concatenate([values[k] for k in rng.integers(0, len(values), len(values))]).mean()
               for _ in range(1000)]
    return {"mean_move_kl_improvement": float(delta[usable].mean()),
            "game_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "games": len(groups)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-reader", type=Path, default=ROOT / "runs/reader-warmup/adapter.pt")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--predictor-steps", type=int, default=400)
    parser.add_argument("--rl-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--bootstrap-texts", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=144)
    args = parser.parse_args()
    if min(args.warmup_steps, args.predictor_steps, args.rl_steps, args.batch_size,
           args.groups, args.bootstrap_texts, args.eval_samples, args.max_tokens) <= 0 or args.candidates < 2:
        parser.error("Counts must be positive and at least two candidates are required")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable here; run with device access or explicitly choose CPU")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    start = time.monotonic()
    logfile = (args.output / "metrics.jsonl").open("w")

    def log(**message):
        message["seconds"] = round(time.monotonic() - start, 3)
        line = json.dumps(message)
        logfile.write(line + "\n")
        logfile.flush()
        print(line, flush=True)

    manifest = json.loads((args.data / "manifest.json").read_text())
    model_lock = json.loads((ROOT / "reader-model.json").read_text())
    configuration = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "prompt": BEHAVIOR_PROMPT, "language_model": model_lock,
                     "initial_reader_sha256": digest(args.initial_reader),
                     "data_manifest": manifest, "torch_version": torch.__version__,
                     "code_sha256": {p: digest(ROOT / p) for p in
                         ["reader/behavior.py", "reader/language.py", "scripts/train_behavior_reader.py"]},
                     "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    train, val = load_data(args.data / "train"), load_data(args.data / "validation")
    train_games = {r["game_id"] for r in train["rows"]}
    val_games = {r["game_id"] for r in val["rows"]}
    if train_games & val_games or manifest["train"]["map_seed"] == manifest["validation"]["map_seed"]:
        raise ValueError("Training/validation split overlap")
    val_idx = np.random.default_rng(405).permutation(len(val["rows"]))[:args.eval_samples]
    model_path = snapshot_download(model_lock["model"], revision=model_lock["revision"], local_files_only=True)
    reader = LanguageReader(model_path, train["activations"].shape[-1], args.device, BEHAVIOR_PROMPT)
    checkpoint = torch.load(args.initial_reader, map_location="cpu", weights_only=True)
    if checkpoint["model"] != model_lock:
        raise ValueError("Initial adapter language model does not match")
    reader.adapter.load_state_dict(checkpoint["adapter"])
    wrapper = BehaviorReader(reader)
    log(stage="start", device=str(device), train_positions=len(train["rows"]),
        player_frozen=True, language_backbone_frozen=True)

    # Labels are generated solely from current visible observations.
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=5e-4)
    for step in range(1, args.warmup_steps + 1):
        idx = rng.integers(0, len(train["rows"]), args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = reader.loss(train["activations"][idx], [train["captions"][i] for i in idx])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite grounding loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            log(stage="english_grounding", step=step, nll=float(loss.detach()))
    before_adapter = copy.deepcopy(reader.adapter.state_dict())
    torch.save({"adapter": before_adapter, "model": model_lock, "prompt": BEHAVIOR_PROMPT},
               args.output / "before_rl.pt")
    reference = copy.deepcopy(reader.adapter).eval().requires_grad_(False)
    bootstrap_idx = rng.permutation(len(train["rows"]))[:args.bootstrap_texts]
    bootstrap_texts = generate_descriptions(wrapper, train["activations"][bootstrap_idx],
                                            8, args.max_tokens, log, "training_descriptions")
    val_texts = generate_descriptions(wrapper, val["activations"][val_idx], 8, args.max_tokens, log,
                                     "validation_descriptions")
    # A single frozen English encoder reads text only. Cache train/validation
    # features; no gradient or player activation can reach a move predictor.
    train_features = wrapper.text_features(train["captions"])
    generated_features = wrapper.text_features(bootstrap_texts)
    mixed_features = train_features.clone()
    mixed_features[bootstrap_idx] = generated_features
    val_features = wrapper.text_features(val_texts)
    models = {}
    for number, name in enumerate(["board_only", "reward_predictor", "independent_auditor"]):
        torch.manual_seed(args.seed + 20 + number)
        model = MovePredictor(train_features.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        best, best_state, best_step = float("inf"), None, None
        for step in range(1, args.predictor_steps + 1):
            idx = rng.integers(0, len(train["rows"]), 64)
            features = mixed_features[idx].clone() if step % 2 else train_features[idx].clone()
            if name == "board_only":
                features.zero_()
            else:
                features[rng.random(len(idx)) < 0.3] = 0  # trains a genuine missing-text control
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs(train, idx, features, device))
            loss = distillation_loss(logits, train["target"][idx].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite predictor loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.predictor_steps:
                vf = torch.zeros_like(val_features) if name == "board_only" else val_features
                pred = predict(model, val, val_idx, vf, device)
                score = float(distillation_loss(pred, val["target"][val_idx]))
                if score < best:
                    best, best_step = score, step
                    best_state = copy.deepcopy(model.state_dict())
                log(stage="fit_predictor", model=name, step=step,
                    train_loss=float(loss.detach()), validation_loss=score)
        model.load_state_dict(best_state)
        model.eval().requires_grad_(False)
        models[name] = model
        torch.save({"model": model.state_dict(), "text_dim": train_features.shape[-1],
                    "best_validation_step": best_step}, args.output / f"{name}.pt")
    before_val, _, _ = evaluate_descriptions(wrapper, models, val, val_idx, val_texts, device)
    (args.output / "validation_before.json").write_text(json.dumps(before_val, indent=2) + "\n")
    log(stage="before_rl", validation=before_val)

    # The reward model is now frozen. Text tokens are genuinely sampled. A
    # leave-one-out baseline compares descriptions for the same board position.
    # There is no differentiable shortcut from activations to predicted moves.
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=2e-4)
    eligible = np.flatnonzero(train["target"][:, :-1].sum(-1).numpy() > 1e-7)
    frozen_hashes = {k: tensor_digest(m) for k, m in models.items()}
    anchor_batches = []
    for step in range(1, args.rl_steps + 1):
        selected = rng.choice(eligible, args.groups, replace=False)
        anchor_batches.append(selected.copy())
        idx = np.repeat(selected, args.candidates)
        hidden = train["activations"][idx]
        tokens, texts = wrapper.sample(hidden, max_new_tokens=args.max_tokens, sample=True)
        features = wrapper.text_features(texts)
        with torch.no_grad():
            pred = predict(models["reward_predictor"], train, idx, features, device)
            empty = predict(models["reward_predictor"], train, idx, torch.zeros_like(features), device)
            target = train["target"][idx]
            cost = behavior_scores(pred, target)["move_ce"]
            empty_cost = behavior_scores(empty, target)["move_ce"]
            rewards = empty_cost - cost
            advantages = (group_advantages(rewards, args.candidates) * 20).clamp(-2, 2).to(device)
            current_adapter = reader.adapter
            reader.adapter = reference
            try:
                ref_logp, _ = wrapper.token_log_probs(hidden, tokens)
            finally:
                reader.adapter = current_adapter
        optimizer.zero_grad(set_to_none=True)
        logp, mask = wrapper.token_log_probs(hidden, tokens)
        policy_loss = -(advantages * (logp * mask).sum(-1)).mean()
        # Nonnegative sampled KL estimator: exp(log p_ref - log p) - ratio - 1.
        ratio = (ref_logp - logp).clamp(-10, 10)
        kl = ((ratio.exp() - ratio - 1) * mask).sum() / mask.sum()
        loss = policy_loss + 0.03 * kl
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite policy-gradient loss")
        loss.backward()
        # Separate backward keeps GPU peak memory bounded.
        anchor = reader.loss(train["activations"][selected], [train["captions"][i] for i in selected])
        (0.15 * anchor).backward()
        grad = torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise RuntimeError("Non-finite adapter gradient")
        optimizer.step()
        log(stage="behavior_rl", step=step, reward_gain=float(rewards.mean()),
            reward_std=float(rewards.std()), policy_loss=float(policy_loss.detach()),
            reference_kl=float(kl.detach()), grounding_nll=float(anchor.detach()),
            gradient_norm=float(grad), unique_descriptions=len(set(texts)))
        with (args.output / "sampled_descriptions.jsonl").open("a") as f:
            for i, text, reward in zip(idx, texts, rewards):
                f.write(json.dumps({"step": step, "train_index": int(i), "text": text,
                                    "reward": float(reward)}) + "\n")
    if any(tensor_digest(models[k]) != v for k, v in frozen_hashes.items()):
        raise AssertionError("A frozen predictor changed during reader RL")
    after_adapter = copy.deepcopy(reader.adapter.state_dict())
    torch.save({"adapter": after_adapter, "model": model_lock, "prompt": BEHAVIOR_PROMPT,
                "configuration": configuration}, args.output / "after_rl.pt")

    # Match the extra visible-fact training without the behavior reward. This
    # distinguishes RL effects from simply spending more time on the warm-start.
    reader.adapter.load_state_dict(before_adapter)
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=2e-4)
    for step, selected in enumerate(anchor_batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = reader.loss(train["activations"][selected], [train["captions"][i] for i in selected])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite anchor-only control loss")
        (0.15 * loss).backward()
        torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            log(stage="anchor_only_control", step=step, grounding_nll=float(loss.detach()))
    anchor_adapter = copy.deepcopy(reader.adapter.state_dict())
    torch.save({"adapter": anchor_adapter, "model": model_lock, "prompt": BEHAVIOR_PROMPT},
               args.output / "anchor_only.pt")

    # Final test is touched for the first time after all fitting and updates.
    test = load_data(args.data / "test")
    test_games = {r["game_id"] for r in test["rows"]}
    if test_games & (train_games | val_games) or manifest["test"]["map_seed"] in {
        manifest["train"]["map_seed"], manifest["validation"]["map_seed"]}:
        raise ValueError("Test split overlap")
    test_idx = np.random.default_rng(406).permutation(len(test["rows"]))[:args.eval_samples]
    reader.adapter.load_state_dict(before_adapter)
    before_texts = generate_descriptions(wrapper, test["activations"][test_idx], 8, args.max_tokens, log,
                                        "test_before_rl")
    before, before_raw, _ = evaluate_descriptions(wrapper, models, test, test_idx, before_texts, device)
    reader.adapter.load_state_dict(after_adapter)
    after_texts = generate_descriptions(wrapper, test["activations"][test_idx], 8, args.max_tokens, log,
                                       "test_after_rl")
    after, after_raw, _ = evaluate_descriptions(wrapper, models, test, test_idx, after_texts, device)
    reader.adapter.load_state_dict(anchor_adapter)
    anchor_texts = generate_descriptions(wrapper, test["activations"][test_idx], 8, args.max_tokens, log,
                                        "test_anchor_only")
    anchor_metrics, anchor_raw, _ = evaluate_descriptions(wrapper, models, test, test_idx, anchor_texts, device)
    reader.adapter.load_state_dict(after_adapter)
    intervals = {}
    for name in ["reward_predictor", "independent_auditor"]:
        intervals[name] = {
            "before_to_after": paired_game_interval(before_raw[name]["real_text"],
                after_raw[name]["real_text"], test["rows"], test_idx),
            "after_real_vs_shuffled": paired_game_interval(after_raw[name]["shuffled_text"],
                after_raw[name]["real_text"], test["rows"], test_idx),
            "after_real_vs_missing": paired_game_interval(after_raw[name]["no_text"],
                after_raw[name]["real_text"], test["rows"], test_idx),
            "rl_vs_anchor_only": paired_game_interval(anchor_raw[name]["real_text"],
                after_raw[name]["real_text"], test["rows"], test_idx)}
    examples = [{"test_index": int(i), **test["rows"][int(i)],
                 "automatic_grounding": test["captions"][int(i)],
                 "before": b, "after": a, "anchor_only": c}
                for i, b, a, c in zip(test_idx, before_texts, after_texts, anchor_texts)]
    report = {"configuration": configuration, "seconds": time.monotonic() - start,
              "test_before_rl": before, "test_after_rl": after, "test_anchor_only": anchor_metrics,
              "paired_comparisons": intervals,
              "examples": examples, "frozen_predictors_verified": True,
              "test_indices": test_idx.tolist(),
              "scope": "Free-form text policy-gradient pilot, with automatic visible-fact grounding. "
                       "Immediate move probabilities only, not future rollouts or causal explanations. "
                       "Auditor has independent predictor weights but shares the frozen English encoder. "
                       "Weak frozen player; a positive reward is not proof of English faithfulness."}
    checkpoint_path = Path(manifest["checkpoint"])
    if digest(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise AssertionError("Original player checkpoint changed")
    report["player_checkpoint_unchanged"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(stage="complete", report=str(args.output / "report.json"),
        paired_comparisons=intervals)
    logfile.close()


def tensor_digest(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
