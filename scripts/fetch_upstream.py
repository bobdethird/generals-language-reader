"""Fetch pinned dependencies, without resetting existing user modifications."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", default="git")
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream-lock.json").read_text())
    (ROOT / "vendor").mkdir(exist_ok=True)
    for name, spec in lock.items():
        destination = ROOT / "vendor" / name
        if not destination.exists():
            subprocess.run([args.git, "clone", "--filter=blob:none", spec["url"], str(destination)], check=True)
            subprocess.run([args.git, "-C", str(destination), "checkout", "--detach", spec["commit"]], check=True)
        actual = subprocess.check_output([args.git, "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output([args.git, "-C", str(destination), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if actual != spec["commit"] or dirty:
            raise RuntimeError(f"{name}: existing checkout differs from lock or has changes; left untouched")
        print(f"{name}: verified {actual}")


if __name__ == "__main__":
    main()
