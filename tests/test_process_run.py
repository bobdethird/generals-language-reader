import os
from pathlib import Path
import sys
import time

from reader.process_run import run_bounded


def test_silent_process_cannot_outrun_limit(tmp_path):
    started = time.monotonic()
    result = run_bounded([sys.executable, "-c", "import time; time.sleep(60)"],
                         cwd=tmp_path, env=os.environ.copy(), log_path=tmp_path / "log", seconds=0.2)
    assert result["timed_out"]
    assert result["exit_code"] != 0
    assert time.monotonic() - started < 5


def test_success_preserves_output_and_exit_status(tmp_path):
    log = tmp_path / "log"
    result = run_bounded([sys.executable, "-c", "print('checkpoint saved')"],
                         cwd=tmp_path, env=os.environ.copy(), log_path=log, seconds=5)
    assert result["exit_code"] == 0
    assert not result["timed_out"]
    assert log.read_text().strip() == "checkpoint saved"


def test_iteration_based_run_has_no_watchdog(tmp_path, monkeypatch):
    def forbidden_timer(*args, **kwargs):
        raise AssertionError("No time limit was requested")
    monkeypatch.setattr("reader.process_run.threading.Timer", forbidden_timer)
    result = run_bounded([sys.executable, "-c", "print('iteration target reached')"],
                         cwd=tmp_path, env=os.environ.copy(), log_path=tmp_path / "log", seconds=None)
    assert result["exit_code"] == 0
    assert not result["timed_out"]
