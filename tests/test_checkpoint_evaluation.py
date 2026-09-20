import json

import pytest

from reader.checkpoint_evaluation import (PROTOCOL_ID, checkpoints, pending_pairs,
                                          read_results, summarize_counts, wandb_metrics)


def test_discovery_uses_confirmed_numbered_ema_files(tmp_path):
    root = tmp_path / "training"
    folder = root / "checkpoints/L_7d_gae90"
    folder.mkdir(parents=True)
    (root / "run.json").write_text('{}')
    (root / "progress.json").write_text('{"iteration": 1000}')
    for name in ("L_7d_gae90_ema_500.eqx", "L_7d_gae90_ema_1000.eqx",
                 "L_7d_gae90_ema_1500.eqx", "L_7d_gae90_1000.eqx",
                 "published_latest_ema.eqx", "L_7d_gae90_ema_264.eqx"):
        (folder / name).write_bytes(b"test")
    assert list(checkpoints(tmp_path, "training")) == [500, 1000]
    with pytest.raises(ValueError):
        checkpoints(tmp_path, "../training")


def test_candidates_use_500_step_anchors_and_do_not_repeat_completed_pairs():
    done = {(500, 500), (1000, 500)}
    assert pending_pairs([2000, 500, 1500, 1000], done) == [
        (1500, 500), (1500, 1000), (2000, 500), (2000, 1000), (2000, 1500)]
    assert pending_pairs([500], done) == []
    assert pending_pairs([500, 600, 700, 1000, 1100], set()) == [
        (600, 500), (700, 500), (1000, 500), (1100, 500), (1100, 1000)]


def test_reference_window_rolls_forward_and_excludes_self():
    values = [3100, 3000, 2500, 2600, 2000, 2100, 1500, 1000, 500, 2500]
    pairs = pending_pairs(values, set())
    assert [ref for candidate, ref in pairs if candidate == 2600] == [1500, 2000, 2500]
    assert [ref for candidate, ref in pairs if candidate == 3000] == [1500, 2000, 2500]
    assert [ref for candidate, ref in pairs if candidate == 3100] == [2000, 2500, 3000]
    assert all(sum(c == step for c, _ in pairs) <= 3 for step in values)


def test_completed_recent_matches_do_not_backfill_older_opponents():
    values = [500, 1000, 1500, 2000, 2500, 2600]
    done = {(2600, 1500), (2600, 2000)}
    assert [pair for pair in pending_pairs(values, done) if pair[0] == 2600] == [(2600, 2500)]
    done.add((2600, 2500))
    assert [pair for pair in pending_pairs(values, done) if pair[0] == 2600] == []


def test_resumed_run_inherits_anchors_but_clips_unselected_parent_history(tmp_path):
    old = tmp_path / "old"
    folder = old / "checkpoints/L_7d_gae90"
    folder.mkdir(parents=True)
    (old / "run.json").write_text('{}')
    (old / "progress.json").write_text('{"iteration": 1000}')
    for step in (500, 600, 1000):
        (folder / f"L_7d_gae90_ema_{step}.eqx").write_bytes(b"test")
    new = tmp_path / "new"
    folder = new / "checkpoints/L_7d_gae90"
    folder.mkdir(parents=True)
    (new / "run.json").write_text('{"resume":{"source":"old","iteration":950}}')
    # Before the first new checkpoint, the old references are still available.
    assert list(checkpoints(tmp_path, "new")) == [500, 600]
    (new / "progress.json").write_text('{"iteration":1100}')
    for step in (1000, 1100):
        (folder / f"L_7d_gae90_ema_{step}.eqx").write_bytes(b"new")
    saved = checkpoints(tmp_path, "new")
    assert list(saved) == [500, 600, 1000, 1100]
    assert saved[1000].is_relative_to(new)


def test_draws_remain_in_win_rate_denominator_and_half_in_score():
    result = summarize_counts(dict(wins=64, losses=32, draws=32))
    assert result["win_rate"] == .5
    assert result["score"] == .625
    assert result["draw_rate"] == .25
    for bad in (dict(wins=0, losses=0, draws=0), dict(wins=-1, losses=1, draws=0)):
        with pytest.raises(ValueError):
            summarize_counts(bad)


def test_control_and_real_matches_have_separate_chart_series():
    result = summarize_counts(dict(wins=32, losses=32, draws=64))
    control = wandb_metrics(dict(candidate_iteration=500, reference_iteration=500, **result))
    real = wandb_metrics(dict(candidate_iteration=1000, reference_iteration=500, **result))
    assert "control/score_vs_500" in control
    assert "checkpoint/score_vs_500" not in control
    assert real["checkpoint/win_rate_vs_500"] == .25
    assert real["iteration"] == 1000


def test_results_ignore_partial_files_and_reject_protocol_mix(tmp_path):
    (tmp_path / "ema-1500-vs-500.tmp").write_text("partial")
    path = tmp_path / "ema-1000-vs-500.json"
    row = dict(protocol_id=PROTOCOL_ID, candidate_iteration=1000, reference_iteration=500)
    path.write_text(json.dumps(row))
    assert read_results(tmp_path) == [row]
    path.write_text(json.dumps({**row, "protocol_id": "different"}))
    with pytest.raises(ValueError, match="Mixed"):
        read_results(tmp_path)
