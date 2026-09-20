"""Download only persisted logs/metrics from a Modal run, without interrupting it."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import modal

FILES = ("run.json", "requested-config.yaml", "training.log", "metrics.jsonl",
         "reference-evaluations/evaluations.jsonl", "reference-evaluations/reference-matrix.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.name):
        parser.error("name must be a simple run directory")
    root = Path(__file__).resolve().parents[1]
    now = dt.datetime.now(dt.timezone.utc)
    destination = root / "runs/snapshots" / now.strftime("%Y%m%dT%H%M%SZ") / args.name
    destination.mkdir(parents=True, exist_ok=False)
    volume = modal.Volume.from_name("generals-policy-checkpoints")
    manifest = dict(source_run=args.name, retrieved_at=now.isoformat(), files={})
    for relative in FILES:
        try:
            data = b"".join(volume.read_file(f"{args.name}/{relative}"))
        except FileNotFoundError:
            if relative in ("run.json", "requested-config.yaml", "training.log"):
                raise
            continue
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"][relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest["retrieval_finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (destination / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(path=str(destination), **manifest), indent=2))


if __name__ == "__main__":
    main()
