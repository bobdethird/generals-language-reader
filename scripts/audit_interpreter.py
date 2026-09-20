"""Independently verify generated checkpoint-3800 explanations against measurements.

Exact agreement with the predeclared strongest-probe target remains the primary
metric. Supported alternative influences and tied strongest probes are reported
separately; neither silently replaces that metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from reader.counterfactuals import verify_rationale, parse_rationale
from reader.decisions import parse_decision, score_decisions
from reader.grounding import paired_game_interval


def audit(directory):
    bundle = json.loads((directory / "bundle.json").read_text())
    for name, record in bundle["files"].items():
        with (directory / name).open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != record["sha256"]:
            raise ValueError(f"Bundle file changed: {name}")
    what = json.loads((directory / "what-report.json").read_text())
    why = json.loads((directory / "why-report.json").read_text())
    rows = [json.loads(line) for line in (directory / "why-test-evidence.jsonl").read_text().splitlines()]
    if len(rows) != len(why["examples"]):
        raise ValueError("Final test evidence and generated outputs differ in length")
    checks, errors = [], []
    for index, (row, example) in enumerate(zip(rows, why["examples"])):
        if row["map_id"] != example["map_id"] or row["why_target"] != example["target"]:
            raise ValueError("Final test records are not aligned")
        check = verify_rationale(example["generated"], row["evidence_audit"])
        check["canonical_match"] = parse_rationale(example["generated"]) == row["rationale"]
        checks.append(check)
        if not check["canonical_match"]:
            errors.append({"index": index, "map_id": row["map_id"], "generated": example["generated"],
                           "target": example["target"], **check})
    groups = [{"game_id": row["map_id"]} for row in rows]
    why_audit = {}
    for key in checks[0]:
        values = np.array([c[key] for c in checks], dtype=float)
        why_audit[key] = paired_game_interval(values[:, None], np.zeros((len(values), 1)), groups)
    exact = why_audit["canonical_match"]["accuracy_gain"]
    if exact != why["test"]["metrics"]["real"]["exact_rationale_accuracy"]:
        raise ValueError("Independent canonical accuracy disagrees with training report")
    what_examples = what["examples"]
    targets = [parse_decision(e["target"]) for e in what_examples]
    if any(t is None for t in targets):
        raise ValueError("Unparseable action target")
    what_metrics, correct = score_decisions([e["generated"] for e in what_examples], targets)
    if what_metrics["exact_action_accuracy"] != what["test"]["metrics"]["real"]["exact_action_accuracy"]:
        raise ValueError("Independent move accuracy disagrees with training report")
    categories = {}
    for name, selection in {
        "wait": [t["pass"] for t in targets],
        "full_moves": [not t["pass"] and not t["half"] for t in targets],
        "half_moves": [not t["pass"] and t["half"] for t in targets],
    }.items():
        selected = np.asarray(selection)
        categories[name] = {"positions": int(selected.sum()),
                            "accuracy": float(correct[selected].mean()) if selected.any() else None}
    why_metrics = why["test"]["metrics"]["real"]
    transfer = json.loads((directory / "transfer-result.json").read_text())
    if transfer["reader"] != bundle["what_reader"] or transfer["player_iteration"] != 3800:
        raise ValueError("Independent observer audited a different reader")
    context = transfer["context_audit"]["readers"][bundle["what_reader"]]
    gates = {
        "move_accuracy_at_least_95_percent": what_metrics["exact_action_accuracy"] >= .95,
        "complete_description_at_least_85_percent": what_metrics["fully_correct_description"] >= .85,
        "rationale_accuracy_at_least_95_percent": exact >= .95,
        "nontrivial_rationale_at_least_95_percent": why_metrics["nontrivial_accuracy"] >= .95,
        "shuffled_evidence_gain_ci_positive": why["test"]["control_gains"]["shuffled_evidence"]["game_bootstrap_95_percent_interval"][0] > 0,
        "frozen_weights_verified": what["frozen_player_unchanged"] and what["frozen_language_unchanged"] and why["language_unchanged"],
        "independent_context_gain_ci_positive": context["movement_accuracy_gain"]["board_only"]["game_bootstrap_95_percent_interval"][0] > 0,
    }
    return {"player_sha256": bundle["player_sha256"], "what": what_metrics,
            "move_categories": categories, "why": why_metrics,
            "independent_why_claim_audit": why_audit,
            "independent_context_audit": transfer["context_audit"],
            "why_errors": errors, "checks": gates, "passed": all(gates.values()),
            "scope": "Local sensitivity to ten predefined input probes. This does not establish strategic intent or future benefit.",
            "evaluation_note": "Population metrics use held-out maps on B200. Local examples are separate checks, not additional population estimates."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("runs/interpreter-3800"))
    args = p.parse_args()
    result = audit(args.bundle)
    (args.bundle / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "why_errors"}, indent=2))
    if not result["passed"]:
        raise SystemExit("Interpreter audit did not meet every gate")


if __name__ == "__main__":
    main()
