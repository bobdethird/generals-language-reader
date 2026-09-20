"""Fixed-protocol checkpoint comparisons, independent of the training process."""
import hashlib
import json
from pathlib import Path
import re

PROTOCOL = {
    "version": 1, "weights": "ema", "games": 512, "maps_per_batch": 64,
    "pool_size": 1024, "pool_seed": 19092601, "match_seed": 19092602,
    "min_grid_size": 17, "max_grid_size": 23, "pad_to": 24,
    "min_generals_distance": 17, "max_generals_distance": 28,
    "castle_val_range": [40, 51], "num_cities_range": [9, 11],
    "mountain_density_range": [0.21, 0.21], "truncation": 2048,
    "action_selection": "greedy", "paired_starting_positions": True,
}
PROTOCOL_ID = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:12]
# Scheduling is independent of the match protocol, so historical results remain valid.
MAX_REFERENCES = 3


def validate_source(source):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", source):
        raise ValueError("Invalid source run")
    return source


def checkpoints(root, source):
    """Confirmed numbered EMA files across the selected checkpoint lineage.

    The mutable 'latest' checkpoint is never an evaluation reference.
    """
    result, seen, ceiling = {}, set(), float("inf")
    while source:
        run = Path(root) / validate_source(source)
        if source in seen:
            raise ValueError("Checkpoint ancestry contains a cycle")
        seen.add(source)
        record = json.loads((run / "run.json").read_text())
        progress_path = run / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"iteration": 0}
        for path in (run / "checkpoints/L_7d_gae90").glob("L_7d_gae90_ema_*.eqx"):
            match = re.fullmatch(r"L_7d_gae90_ema_(\d+)\.eqx", path.name)
            if match:
                iteration = int(match[1])
                if iteration >= 500 and iteration % 100 == 0 and iteration <= min(ceiling, progress["iteration"]):
                    result.setdefault(iteration, path)
        resume = record.get("resume", {})
        source = resume.get("source")
        ceiling = min(ceiling, resume.get("iteration", ceiling))
    return dict(sorted(result.items()))


def pending_pairs(iterations, completed):
    """Evaluate candidates against their three most recent earlier 500-step anchors."""
    values = sorted(set(iterations))
    anchors = [step for step in values if step % 500 == 0]
    result = []
    for candidate in values:
        references = [step for step in anchors if step < candidate][-MAX_REFERENCES:]
        # Select the window BEFORE skipping completed pairs; never backfill older opponents.
        result.extend((candidate, reference) for reference in references
                      if (candidate, reference) not in completed)
    return result


def result_path(root, candidate, reference):
    return Path(root) / f"ema-{candidate}-vs-{reference}.json"


def read_results(root):
    results = []
    for path in sorted(Path(root).glob("ema-*-vs-*.json")):
        row = json.loads(path.read_text())
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError("Mixed evaluation protocols")
        results.append(row)
    return sorted(results, key=lambda row: (row["candidate_iteration"], row["reference_iteration"]))


def summarize_counts(counts):
    if set(counts) != {"wins", "losses", "draws"} or any(
            not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError("Invalid game outcomes")
    total = sum(counts.values())
    if not total:
        raise ValueError("No completed games")
    return {**counts, "games": total, "win_rate": counts["wins"] / total,
            "loss_rate": counts["losses"] / total, "draw_rate": counts["draws"] / total,
            "score": (counts["wins"] + counts["draws"] / 2) / total}


def wandb_metrics(row):
    prefix = "control" if row["candidate_iteration"] == row["reference_iteration"] else "checkpoint"
    reference = row["reference_iteration"]
    return {"iteration": row["candidate_iteration"],
            "match_id": f"{row['candidate_iteration']}-vs-{reference}",
            **{f"{prefix}/{field}_vs_{reference}": row[field]
               for field in ("win_rate", "draw_rate", "loss_rate", "score", "games")}}
