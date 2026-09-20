"""Bounded local warm-start; neither game weights nor language weights change.

This establishes an activation-to-English interface, not faithful explanations
or an NLA reconstruction experiment. Validation includes shuffled/zero inputs.
"""
from __future__ import annotations
import argparse
import datetime as dt
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


def load_split(path):
    with np.load(path / "samples.npz", allow_pickle=False) as arrays:
        hidden = np.array(arrays["activations"], dtype=np.float32)
    rows = [json.loads(line) for line in (path / "captions.jsonl").read_text().splitlines()]
    if len(rows) != len(hidden):
        raise ValueError("Activation and caption counts differ")
    return hidden, rows


@torch.no_grad()
def evaluate(reader, hidden, rows, batch_size):
    rng = np.random.default_rng(718)
    shuffled = hidden[rng.permutation(len(hidden))]
    result = {}
    for condition, inputs in [("real", hidden), ("shuffled", shuffled), ("zero", np.zeros_like(hidden))]:
        total, count = 0.0, 0
        for start in range(0, len(hidden), batch_size):
            captions = [row["caption"] for row in rows[start:start + batch_size]]
            total += float(reader.loss(inputs[start:start + batch_size], captions)) * len(captions)
            count += len(captions)
        result[f"{condition}_nll"] = total / count
    result["shuffle_minus_real_nll"] = result["shuffled_nll"] - result["real_nll"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "runs/activations-pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reader-warmup")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_samples) <= 0:
        parser.error("steps, batch-size and eval-samples must be positive")
    torch.manual_seed(44)
    torch.set_num_threads(6)
    rng = np.random.default_rng(44)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; explicitly select --device cpu for a CPU run")
    model_lock = json.loads((ROOT / "reader-model.json").read_text())
    model_path = snapshot_download(model_lock["model"], revision=model_lock["revision"], local_files_only=True)
    train_h, train_rows = load_split(args.data / "train")
    val_h, val_rows = load_split(args.data / "validation")
    train_games = {row["game_id"] for row in train_rows}
    if train_games & {row["game_id"] for row in val_rows}:
        raise ValueError("Train and validation games overlap")
    chosen = np.random.default_rng(404).permutation(len(val_h))[:args.eval_samples]
    val_h, val_rows = val_h[chosen], [val_rows[i] for i in chosen]
    args.output.mkdir(parents=True, exist_ok=False)
    reader = LanguageReader(model_path, train_h.shape[-1], args.device)
    optimizer = torch.optim.AdamW(reader.adapter.parameters(), lr=args.lr)
    print(f"LOCAL {args.device}: {args.steps} warm-start updates; only adapter parameters train", flush=True)
    print(f"Adapter parameters: {sum(p.numel() for p in reader.adapter.parameters()):,}", flush=True)
    started = time.monotonic()
    before = evaluate(reader, val_h, val_rows, args.batch_size)
    print(f"Initial validation: {before}", flush=True)
    with (args.output / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            indices = rng.integers(0, len(train_h), args.batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = reader.loss(train_h[indices], [train_rows[i]["caption"] for i in indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.adapter.parameters(), 1.0)
            optimizer.step()
            metric = {"step": step, "train_nll": float(loss.detach()), "seconds": time.monotonic() - started}
            log.write(json.dumps(metric) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0:
                print(metric, flush=True)
    after = evaluate(reader, val_h, val_rows, args.batch_size)
    checkpoint = {"adapter": reader.adapter.state_dict(), "activation_dim": train_h.shape[-1],
                  "model": model_lock, "prompt_version": 1,
                  "data_manifest": json.loads((args.data / "manifest.json").read_text())}
    torch.save(checkpoint, args.output / "adapter.pt")
    examples = []
    for i in range(min(8, len(val_h))):
        generated = reader.generate(val_h[i:i + 1])[0]
        control = reader.generate(val_h[(i + 1) % len(val_h)][None])[0]
        examples.append({**val_rows[i], "generated": generated,
                         "generated_with_other_activation": control})
        print(f"Example {i}: {generated}", flush=True)
    report = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "device": args.device,
              "steps": args.steps, "seconds": time.monotonic() - started,
              "eval_samples": len(val_h), "before": before, "after": after,
              "model": model_lock, "examples": examples,
              "scope": "Observable-fact warm-start only; no reconstruction RL or causal explanation claims."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Final validation: {after}\nSaved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
