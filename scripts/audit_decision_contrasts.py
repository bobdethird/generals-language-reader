"""Audit frozen readers on same-fact positions with different player decisions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from reader.decision_data import load_decision_split
from reader.decision_factory import restore_reader
from reader.decision_audit import matched_decision_pairs, decision_probability_audit
from reader.decisions import score_decisions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--readers", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=128)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Contrast generation requires the requested GPU")
    torch.set_num_threads(8)
    data = load_decision_split(args.data, "test", include_mask=True)
    pairs = matched_decision_pairs(data, limit=args.pairs)
    if not pairs:
        raise ValueError("No matched decision contrasts found")
    selected = sorted({i for pair in pairs for i in pair})
    lookup = {index: offset for offset, index in enumerate(selected)}
    results = {"pairs": len(pairs), "positions": len(selected), "maps": len(pairs), "readers": {},
               "scope": "Observational contrasts on the same map with identical coarse board facts and different "
                        "greedy decisions. These test decision-specific language, not a causal intervention."}
    for path in args.readers:
        checkpoint_path = path / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reader = restore_reader(checkpoint, "cuda")
        hidden = data["hidden"][selected]
        if not reader.legal_mask:
            hidden = hidden[..., :448]
        texts = []
        for start in range(0, len(hidden), 64):
            texts.extend(reader.generate(hidden[start:start+64], max_new_tokens=90))
        targets = [data["decision_facts"][i] for i in selected]
        metrics, correct = score_decisions(texts, targets)
        both_correct = [bool(correct[lookup[a]] and correct[lookup[b]]) for a, b in pairs]
        result = {"checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "selected_step": checkpoint["step"], "metrics": metrics,
                  "both_decisions_correct": float(np.mean(both_correct)),
                  "probability_audit": decision_probability_audit(texts, data["logits"][selected]),
                  "examples": [{"map_id": data["rows"][a]["map_id"], "first_index": a, "second_index": b,
                      "first_target": data["captions"][a], "second_target": data["captions"][b],
                      "first_text": texts[lookup[a]], "second_text": texts[lookup[b]], "both_correct": ok}
                      for (a, b), ok in zip(pairs, both_correct)]}
        results["readers"][path.name] = result
        print(json.dumps({"stage": "contrast_audit", "reader": path.name, "metrics": metrics,
                          "both_decisions_correct": result["both_decisions_correct"]}), flush=True)
        del reader, checkpoint
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == "__main__":
    main()
