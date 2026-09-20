"""Generate a decision description and verify it against a saved player snapshot.

Readers see activations, optionally legality and a frozen original action head.
Saved player outputs and observations are used AFTER generation for an independent
check; they never repair the raw text.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from reader.decision_factory import restore_reader
from reader.decision_data import pack_legal_masks
from reader.decisions import (greedy_actions, decision_facts, decode_action,
                              parse_decision, score_decisions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA GPU unavailable")
    checkpoint = torch.load(args.reader, map_location="cpu", weights_only=True)
    torch.set_num_threads(8)
    reader = restore_reader(checkpoint, args.device)
    with np.load(args.samples, allow_pickle=False) as arrays:
        if not 0 <= args.index < len(arrays["activations"]):
            raise ValueError("Snapshot index outside dataset")
        hidden = arrays["activations"][args.index:args.index + 1]
        if reader.legal_mask:
            hidden = pack_legal_masks(hidden, arrays["masks"][args.index:args.index + 1])
        observation = arrays["observations"][args.index]
        logits = arrays["logits"][args.index:args.index + 1]
        sampled = arrays["actions"][args.index].tolist()
    text = reader.generate(hidden, max_new_tokens=90)[0]
    greedy = int(greedy_actions(logits)[0])
    facts = decision_facts(observation, greedy)
    metrics, _ = score_decisions([text], [facts])
    result = {"raw_reader_text": text, "decoded_reader_action": parse_decision(text),
              "verified_supported_description": metrics["fully_correct_description"] == 1,
              "verified_text": text if metrics["fully_correct_description"] == 1 else None,
              "original_player_greedy_action": decode_action(greedy),
              "rollout_sampled_action": sampled, "verification": metrics,
              "reader_variant": checkpoint["variant"], "reader_step": checkpoint["step"],
              "snapshot_index": args.index,
              "note": "Verification checks a decision description, not a recovered strategic motive. "
                      "Coordinates in text are one-based; coordinates in JSON actions are zero-based. "
                      "A failed description remains unverified; the original action is shown separately."}
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
