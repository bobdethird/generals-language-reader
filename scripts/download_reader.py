"""Explicit, revision-pinned download of public reader weights (~1.2 GB)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from huggingface_hub import HfApi, snapshot_download

MODEL = "Qwen/Qwen3-0.6B"
lock = ROOT / "reader-model.json"
revision = (json.loads(lock.read_text())["revision"] if lock.exists()
            else HfApi().model_info(MODEL, token=False).sha)
path = snapshot_download(MODEL, revision=revision, token=False,
                         allow_patterns=["*.json", "*.safetensors", "*.txt"])
lock.write_text(json.dumps({"model": MODEL, "revision": revision}, indent=2) + "\n")
print(f"Downloaded {MODEL}@{revision} to {path}")
