import json
import pytest

from reader.wandb_sync import complete_rows, events, lineage, public_config


def write_run(root, name, rows, timings=(), resume=None):
    folder = root / name
    folder.mkdir()
    record = dict(name=name, status="running", gpu="B200", gpu_count=8,
                  global_envs=2048, global_minibatch=4096)
    if resume:
        record["resume"] = resume
    (folder / "run.json").write_text(json.dumps(record))
    for filename, values in (("metrics.jsonl", rows), ("iteration-timing.jsonl", timings)):
        (folder / filename).write_text("".join(json.dumps(row) + "\n" for row in values))
    return folder


def timing(iteration, timestamp):
    return dict(iteration=iteration, timestamp=timestamp, stage=2,
                update_seconds=4, rollout_seconds=2, ppo_seconds=2, samples=200)


def test_checkpoint_ancestry_excludes_siblings_and_post_checkpoint_updates(tmp_path):
    write_run(tmp_path, "parent", [{"step": n, "train/loss": n} for n in (1, 2, 3)])
    write_run(tmp_path, "sibling", [{"step": 3, "train/loss": 999}])
    write_run(tmp_path, "current", [{"step": n, "train/loss": n} for n in (2, 3, 4)],
              resume={"source": "parent", "iteration": 2})
    segments = lineage(tmp_path, "current")
    result = list(events(tmp_path, segments))
    assert [r["iteration"] for r in result] == [1, 2, 3, 4]
    assert [r["source_run"] for r in result] == ["parent", "parent", "current", "current"]


def test_duplicate_steps_delayed_eval_partial_writes_and_restart_ids(tmp_path):
    folder = write_run(tmp_path, "current", [
        {"step": 1, "train/loss": 3}, {"step": 1, "eval/win_rate": .5},
        {"step": 1, "eval/won_both": .2}], [timing(1, 100)])
    segments = lineage(tmp_path, "current")
    first = list(events(tmp_path, segments))
    assert len(first) == 4
    seen = {row["sync/event_id"] for row in first}
    path = folder / "metrics.jsonl"
    with path.open("a") as file:
        file.write('{"step": 2, "train/loss": 2')
    assert len(list(events(tmp_path, segments))) == 4
    with path.open("a") as file:
        file.write('}\n{"step": 1, "eval/draw_rate": 0.1}\n')
    new = [r for r in events(tmp_path, segments) if r["sync/event_id"] not in seen]
    assert len(new) == 2
    assert new[0]["eval/draw_rate"] == .1
    assert new[1]["train/loss"] == 2
    assert len({r["sync/event_id"] for r in events(tmp_path, segments)}) == 6


def test_rates_include_pauses_without_bridging_missing_iterations(tmp_path):
    write_run(tmp_path, "current", [], [timing(1, 100), timing(2, 120), timing(4, 126)])
    result = list(events(tmp_path, lineage(tmp_path, "current")))
    assert "performance/iterations_per_second" not in result[0]
    assert result[1]["performance/player_samples_per_second"] == 10
    assert result[1]["performance/iterations_per_second"] == .05
    assert "performance/iterations_per_second" not in result[2]
    assert result[1]["curriculum/distance_max"] == 13


def test_nonfinite_metrics_are_reported_and_metadata_allowlisted(tmp_path):
    folder = write_run(tmp_path, "current", [{"step": 1, "train/loss": float("nan"),
                                              "train/entropy": 4, "unrelated": "private"}])
    record = json.loads((folder / "run.json").read_text())
    record["secret"] = "must-never-upload"
    (folder / "run.json").write_text(json.dumps(record))
    segments = lineage(tmp_path, "current")
    row = next(events(tmp_path, segments))
    assert "train/loss" not in row and "unrelated" not in row
    assert row["sync/nonfinite_count"] == 1
    assert row["sync/nonfinite_metrics"] == "train/loss"
    assert "must-never-upload" not in json.dumps(public_config(segments))


def test_completed_corruption_and_ancestry_cycles_fail_loudly(tmp_path):
    folder = write_run(tmp_path, "current", [], resume={"source": "current", "iteration": 1})
    with pytest.raises(ValueError, match="cycle"):
        lineage(tmp_path, "current")
    (folder / "metrics.jsonl").write_text('{bad}\n')
    with pytest.raises(json.JSONDecodeError):
        list(complete_rows(folder / "metrics.jsonl"))
    with pytest.raises(ValueError, match="Invalid source"):
        lineage(tmp_path, "../somewhere")
