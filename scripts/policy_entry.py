"""Compatibility entry point; normal upstream CLI arguments pass through."""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader.runtime import import_upstream, install_jax_compat, install_simulator_compat

import_upstream()
install_jax_compat()
install_simulator_compat()
if os.environ.get("AVERAGEJOE_CURRICULUM_SUPPORT") == "1":
    from reader.upstream_training import install_training_support
    install_training_support()
if os.environ.get("AVERAGEJOE_PUBLISHED_SUPPORT") == "1":
    from reader.published_training import install_published_support
    install_published_support()
runpy.run_path(str(ROOT / "vendor/averagejoe/main.py"), run_name="__main__")
