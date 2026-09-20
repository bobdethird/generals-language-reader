"""Run a bounded local policy experiment, preserving logs and provenance."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local-smoke.yaml")
    parser.add_argument("--name", default="local-smoke")
    args = parser.parse_args()
    config = args.config.resolve()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name")
    output = ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1", WANDB_MODE="disabled",
               MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
               XDG_CACHE_HOME=str(ROOT / ".cache"))
    command = [sys.executable, "-u", str(ROOT / "scripts/policy_entry.py"),
               "--config", str(config)]
    record = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "location": "local", "device": "cpu", "command": command,
              "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Local CPU run; log: {output / 'training.log'}", flush=True)
    with (output / "training.log").open("w") as log:
        process = subprocess.Popen(command, cwd=output, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            result = process.wait()
        record.update(exit_code=result, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
