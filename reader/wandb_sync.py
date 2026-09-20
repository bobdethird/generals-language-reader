"""Scalar-only, restartable conversion of append-only learner logs into W&B events."""
import hashlib
import json
import math
import re
from pathlib import Path

DISTANCES = ((2, 6), (4, 8), (6, 13), (11, 17), (17, 28))


def valid_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
        raise ValueError("Invalid source run name")
    return name


def lineage(root, name):
    """Follow checkpoint parents, clipping each ancestor at the selected checkpoint."""
    result, seen, upper = [], set(), None
    while name:
        valid_name(name)
        if name in seen:
            raise ValueError("Checkpoint ancestry contains a cycle")
        seen.add(name)
        record = json.loads((Path(root) / name / "run.json").read_text())
        resume = record.get("resume", {})
        lower = int(resume.get("iteration", -1))
        if upper is not None and lower > upper:
            raise ValueError("Checkpoint ancestry moves backwards")
        result.append(dict(name=name, record=record, lower=lower, upper=upper))
        name, upper = resume.get("source"), lower
    return list(reversed(result))


def complete_rows(path):
    """Ignore an unfinished final write; reject corruption in completed lines."""
    if not path.exists():
        return
    for index, raw in enumerate(path.read_bytes().splitlines(keepends=True)):
        if not raw.endswith(b"\n"):
            break
        if raw.strip():
            yield index, raw, json.loads(raw)


def event_id(name, filename, index, raw):
    prefix = f"{name}/{filename}/{index}:".encode()
    return hashlib.sha256(prefix + raw).hexdigest()[:32]


def finite_metrics(row):
    values, invalid = {}, []
    for key, value in row.items():
        if not key.startswith(("train/", "eval/")):
            continue
        if isinstance(value, (float, int)) and math.isfinite(value):
            values[key] = value
        else:
            invalid.append(key)
    values["sync/nonfinite_count"] = len(invalid)
    if invalid:
        values["sync/nonfinite_metrics"] = ",".join(sorted(invalid))
    return values


def events(root, segments):
    """Keep repeated-step eval rows; use a custom iteration axis, not W&B's step.

    Immutable per-line IDs let a restarted mirror deduplicate against server
    history. Timing and metric writes can arrive in either order without loss.
    """
    for segment in segments:
        name, record = segment["name"], segment["record"]
        pending = []
        for filename in ("metrics.jsonl", "iteration-timing.jsonl"):
            previous = None
            for index, raw, row in complete_rows(Path(root) / name / filename):
                iteration = int(row["step"] if filename == "metrics.jsonl" else row["iteration"])
                if iteration <= segment["lower"]:
                    previous = row
                    continue
                if segment["upper"] is not None and iteration > segment["upper"]:
                    continue
                values = {"iteration": iteration, "source_run": name,
                          "sync/event_id": event_id(name, filename, index, raw)}
                if filename == "metrics.jsonl":
                    values.update(finite_metrics(row))
                else:
                    stage = int(row["stage"])
                    values.update({
                        "curriculum/stage": stage,
                        "performance/update_seconds": row["update_seconds"],
                        "performance/rollout_seconds": row["rollout_seconds"],
                        "performance/ppo_seconds": row["ppo_seconds"],
                        "performance/player_samples_per_iteration": row["samples"],
                        "performance/gpu_count": record["gpu_count"],
                        "performance/global_envs": record["global_envs"],
                        "performance/global_minibatch": record["global_minibatch"],
                        "timing/learner_timestamp": row["timestamp"],
                    })
                    if 0 <= stage < len(DISTANCES):
                        values["curriculum/distance_min"], values["curriculum/distance_max"] = DISTANCES[stage]
                    # Consecutive timestamps include evaluation/checkpoint/compile pauses.
                    # Never bridge process restarts or infer a rate across missing updates.
                    if previous and iteration == int(previous["iteration"]) + 1:
                        elapsed = row["timestamp"] - previous["timestamp"]
                        if elapsed > 0:
                            values["performance/iteration_wall_seconds"] = elapsed
                            values["performance/iterations_per_second"] = 1 / elapsed
                            values["performance/player_samples_per_second"] = row["samples"] / elapsed
                    memory = row.get("gpu_memory", [])
                    if memory:
                        values["performance/max_gpu_allocated_gib"] = max(
                            item.get("bytes_in_use", 0) for item in memory) / 2**30
                    previous = row
                pending.append((iteration, filename, index, values))
        for _, _, _, values in sorted(pending, key=lambda item: item[:3]):
            yield values


def public_config(segments):
    """Only explicitly selected, non-secret training metadata leaves Modal."""
    fields = ("gpu", "gpu_count", "global_envs", "global_minibatch", "rollout_steps",
              "batch_mode", "config_sha256", "started_at", "upstream")
    return {"source_run": segments[-1]["name"],
            "target_iterations": segments[-1]["record"].get("target_iterations"),
            "deadline_unix": segments[-1]["record"].get("deadline_unix"),
            "metric_origin": "Saved learner logs; uploader system metrics disabled",
            "evaluation": "eval/win_rate is versus random play; train/win_rate is self-play seat outcomes",
            "lineage": [{"name": item["name"], "after_iteration": item["lower"],
                         "through_iteration": item["upper"],
                         **{key: item["record"][key] for key in fields if key in item["record"]}}
                        for item in segments]}
