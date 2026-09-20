"""Audit saved generated text by action type so common full moves cannot hide errors."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from reader.decisions import parse_decision, encode_action, score_decisions
from reader.grounding import paired_game_interval


def summarize(report):
    examples = report["examples"]
    targets = [parse_decision(e["target"]) for e in examples]
    if any(t is None for t in targets):
        raise ValueError("Unparseable automatic target")
    texts = [e["generated"] for e in examples]
    predictions = [parse_decision(t) for t in texts]
    overall, correct = score_decisions(texts, targets)
    rows = [{"game_id": e["map_id"]} for e in examples]
    if len({r["game_id"] for r in rows}) < 2:
        raise ValueError("Need multiple held-out maps")
    interval = paired_game_interval(correct[:, None], np.zeros((len(correct), 1)), rows)
    groups = {
        "wait": lambda t: t["pass"],
        "full_moves": lambda t: not t["pass"] and not t["half"],
        "half_moves": lambda t: not t["pass"] and t["half"],
        **{d: (lambda t, direction=d: not t["pass"] and t["direction"] == direction)
           for d in ("north", "south", "west", "east")},
        "into_visible_enemy": lambda t: not t["pass"] and t["destination"] == "enemy",
    }
    breakdown = {}
    for name, condition in groups.items():
        selected = [i for i, target in enumerate(targets) if condition(target)]
        breakdown[name] = {"positions": len(selected),
                           "maps": len({rows[i]["game_id"] for i in selected}),
                           "exact_accuracy": float(correct[selected].mean()) if selected else None,
                           "fully_correct_description": score_decisions([texts[i] for i in selected],
                               [targets[i] for i in selected])[0]["fully_correct_description"] if selected else None}
    missed_half = sum(not t["pass"] and t["half"] and
                     (p is None or p["pass"] or not p["half"]) for t, p in zip(targets, predictions))
    return {"selected_step": report["best_step"], "overall": overall,
            "exact_accuracy_map_bootstrap_95_percent_interval": interval["game_bootstrap_95_percent_interval"],
            "maps": interval["games"], "by_action_type": breakdown,
            "half_moves_described_as_other_amount_or_missing": missed_half,
            "frozen_head_test": report.get("frozen_head_test"),
            "frozen_player_unchanged": report["frozen_player_unchanged"],
            "frozen_language_unchanged": report["frozen_language_unchanged"],
            "scope": "Decision and visible-context fidelity. Small action subgroups carry substantial uncertainty."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    results = {path.parent.name: summarize(json.loads(path.read_text())) for path in args.reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
