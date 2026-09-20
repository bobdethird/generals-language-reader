import json

import pytest

from reader.checkpoint_evaluation import (PROTOCOL_ID, checkpoints, pending_pairs,
                                          read_results, summarize_counts, wandb_metrics)


def test_discovery_uses_confirmed_numbered_ema_files(tmp_path):
    root = tmp_path / "training"
    folder = root / "checkpoints/L_7d_gae90"
    folder.mkdir(parents=True)
    (root / "progress.json").write_text('{"iteration": 1000}')
    for name in ("L_7d_gae90_ema_500.eqx", "L_7d_gae90_ema_1000.eqx",
                 "L_7d_gae90_ema_1500.eqx", "L_7d_gae90_1000.eqx",
                 "published_latest_ema.eqx", "L_7d_gae90_ema_264.eqx"):
        (folder / name).write_bytes(b"test")
    assert list(checkpoints(tmp_path, "training")) == [500, 1000]
    with pytest.raises(ValueError):
        checkpoints(tmp_path, "../training")


def test_future_checkpoints_keep_old_anchors_and_do_not_repeat_completed_pairs():
    done = {(500, 500), (1000, 500)}
    assert pending_pairs([2000, 500, 1500, 1000], done) == [
        (1500, 500), (1500, 1000), (2000, 500), (2000, 1000), (2000, 1500)]
    assert pending_pairs([500], done) == []


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
