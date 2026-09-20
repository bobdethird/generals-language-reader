"""NVIDIA GPU training to the configured iteration target; artifacts persist in a Volume.

Run with the isolated client: .venv-modal/bin/modal run modal_policy.py --preset smoke
Only whitelisted project code is uploaded; vendor repos are fetched by pinned SHA.
No local runs, cached weights, credentials, or unrelated workspace files upload.
"""
from pathlib import Path, PurePosixPath
import datetime as dt
import json
import re
import sys

import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = "generals-policy-training"
VOLUME_NAME = "generals-policy-checkpoints"
PLATFORM_TIMEOUT_SECONDS = 24 * 60 * 60  # Modal's maximum per function invocation.
GPU_TYPES = ("H100", "H100:2", "H200", "B200", "B300")  # H100 minimum.
PRESET_CONFIGS = {"smoke": "local-smoke.yaml", "pilot": "modal-pilot.yaml",
                  "continue": "modal-continue.yaml",
                  "curriculum": "modal-curriculum.yaml",
                  "wide": "modal-wide.yaml", "wide-batch": "modal-wide-batch.yaml",
                  "wide-short": "modal-wide-short.yaml",
                  "dual": "modal-dual.yaml"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(ROOT / "requirements-cloud.txt"))
    .add_local_file(ROOT / "requirements-cloud.txt", "/project/requirements-cloud.txt", copy=True)
    .add_local_file(ROOT / "upstream-lock.json", "/project/upstream-lock.json", copy=True)
    .add_local_file(ROOT / "scripts/fetch_upstream.py", "/project/scripts/fetch_upstream.py", copy=True)
    .run_commands("python /project/scripts/fetch_upstream.py",
                  "pip install --no-deps /project/vendor/generals-bots")
    .env({"PYTHONPATH": "/project", "JAX_PLATFORMS": "cuda", "WANDB_MODE": "disabled",
          "PYTHONUNBUFFERED": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
          "JAX_COMPILATION_CACHE_DIR": "/runs/.jax-cache",
          "MPLCONFIGDIR": "/tmp/matplotlib", "SDL_VIDEODRIVER": "dummy"})
    .add_local_dir(ROOT / "reader", "/project/reader", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", "/project/scripts", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "configs", "/project/configs")
    .add_local_file(ROOT / "modal_policy.py", "/project/modal_policy.py")
)
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def validate_run(preset, name, seconds):
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Preset must be one of {tuple(PRESET_CONFIGS)}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Run name must be a simple directory name")
    if seconds < 0:
        raise ValueError("seconds must be zero (no application time limit) or positive")


FUNCTION_OPTIONS = dict(image=image, cpu=(2, 2), memory=(8192, 16384),
                        volumes={"/runs": volume}, timeout=PLATFORM_TIMEOUT_SECONDS, startup_timeout=180,
                        retries=0, min_containers=0, max_containers=1, scaledown_window=2,
                        single_use_containers=True, include_source=False)


def run_training(preset: str, name: str, seconds: int, gpu: str, gpu_count: int = 1):
    import hashlib
    import importlib.metadata as metadata
    import os
    import subprocess
    import time

    validate_run(preset, name, seconds)
    output = Path("/runs") / name
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = {"preset": preset, "name": name, "gpu_requested": gpu,
              "gpu_count": gpu_count,
              "runtime_limit_seconds": seconds or None,
              "platform_timeout_seconds": PLATFORM_TIMEOUT_SECONDS,
              "stop_condition": "configured_iterations" if seconds == 0 else "iterations_or_requested_time_limit",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        import jax
        if not jax.devices() or any(d.platform != "gpu" for d in jax.devices()):
            raise RuntimeError(f"GPU required, got {jax.devices()}")
        if jax.device_count() != gpu_count:
            raise RuntimeError(f"Expected {gpu_count} GPUs, got {jax.devices()}")
        record["jax_devices"] = [str(d) for d in jax.devices()]
        record["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
        if any(gpu not in line for line in record["nvidia_smi"].splitlines()):
            raise RuntimeError(f"Requested {gpu}, received {record['nvidia_smi']}")
        print(f"Hardware: {record['nvidia_smi']}", flush=True)
        record["dependencies"] = {d.metadata["Name"]: d.version for d in metadata.distributions()}
        record["upstream"] = json.loads(Path("/project/upstream-lock.json").read_text())
        source = Path("/project/configs") / PRESET_CONFIGS[preset]
        config = output / "requested-config.yaml"
        config.write_bytes(source.read_bytes())
        record["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        if preset in ("continue", "curriculum"):
            from ruamel.yaml import YAML
            cfg = YAML(typ="safe").load(config.read_text())
            resume = json.loads(Path(f"/project/configs/modal-{preset}-source.json").read_text())
            if cfg["iteration_offset"] != resume["source_iteration"]:
                raise ValueError("Resume schedule offset does not match source iteration")
            for field in ("init_checkpoint", "ema_checkpoint"):
                expected = resume[field]
                if cfg[field] != expected["path"]:
                    raise ValueError(f"Unexpected resume path for {field}")
                digest = hashlib.sha256(Path(cfg[field]).read_bytes()).hexdigest()
                if digest != expected["sha256"]:
                    raise ValueError(f"Resume checkpoint checksum mismatch: {field}")
            resume["target_cumulative_iteration"] = cfg["iteration_offset"] + cfg["num_iters"]
            resume["rollout_seed"] = cfg["seed"]
            record["resume"] = resume
            print(f"Verified resume checkpoints at iteration {resume['source_iteration']}; "
                  f"target cumulative iteration {resume['target_cumulative_iteration']}", flush=True)
            if resume.get("references"):
                import shutil
                ref_dir = output / cfg["ref_eval_dir"]
                ref_dir.mkdir()
                for ref in resume["references"]:
                    for field, suffix in (("checkpoint", ".eqx"), ("config", ".yaml")):
                        source_path = Path(ref[field]["path"])
                        if hashlib.sha256(source_path.read_bytes()).hexdigest() != ref[field]["sha256"]:
                            raise ValueError(f"Reference {field} checksum mismatch: {ref['name']}")
                        shutil.copy2(source_path, ref_dir / (ref["name"] + suffix))
                # Build the matrix in the same fixed environment as the candidate tests.
                (ref_dir / "h2h.json").write_text('{"h2h": {}}\n')
                print(f"Verified and staged {len(resume['references'])} frozen reference opponents.", flush=True)
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        from reader.process_run import run_bounded
        command = [sys.executable, "-u", "/project/scripts/policy_entry.py", "--config", str(config)]
        child_env = os.environ.copy()
        if preset == "curriculum":
            child_env["AVERAGEJOE_CURRICULUM_SUPPORT"] = "1"
            record["training_support_sha256"] = hashlib.sha256(
                Path("/project/reader/upstream_training.py").read_bytes()).hexdigest()
        # Each benchmark starts with an empty persistent compilation cache.
        # /tmp also avoids downloading compiler binaries with model artifacts.
        child_env["JAX_COMPILATION_CACHE_DIR"] = f"/tmp/jax-cache-{name}"
        record["compilation_cache"] = "fresh per-run temporary directory"
        # No CPU fallback: the child inherits JAX_PLATFORMS=cuda.
        remaining = max(0.1, seconds - (time.monotonic() - started)) if seconds else None
        record.update(run_bounded(command, cwd=output, env=child_env,
                                  log_path=output / "training.log", seconds=remaining))
        record["status"] = ("time_limit" if record["timed_out"] else
                            "completed" if record["exit_code"] == 0 else "failed")
        record["checkpoints"] = [str(p.relative_to(output)) for p in sorted(output.rglob("*.eqx"))]
        if preset == "curriculum":
            stages = re.findall(r"CURRICULUM stage (\d+)/\d+ \(iter (\d+),.*?dist=(\d+)-(\d+)",
                                (output / "training.log").read_text())
            record["curriculum_transitions"] = [dict(stage=int(s), local_iteration=int(i),
                min_generals_distance=int(lo), max_generals_distance=int(hi)) for s, i, lo, hi in stages]
            record["final_curriculum_stage"] = int(stages[-1][0]) if stages else 0
        remaining = seconds - (time.monotonic() - started) if seconds else None
        if preset != "smoke" and record["status"] == "completed" and (remaining is None or remaining >= 60):
            candidates = list(output.rglob("*_ema_[0-9]*.eqx"))
            if candidates:
                checkpoint = max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
                evaluation = [sys.executable, "-u", "/project/scripts/evaluate_player.py",
                              "--config", str(config), "--checkpoint", str(checkpoint),
                              "--output", str(output / "heldout-evaluation.json"), "--games", "128"]
                if preset == "curriculum":
                    evaluation.extend(["--curriculum-stage", str(record["final_curriculum_stage"])])
                record["heldout_evaluation_process"] = run_bounded(
                    evaluation, cwd=output, env=child_env,
                    log_path=output / "evaluation.log", seconds=remaining)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        volume.commit()
    print(json.dumps({k: v for k, v in record.items() if k != "dependencies"}, indent=2), flush=True)
    return record


@app.function(gpu="H100!", **FUNCTION_OPTIONS)
def train_h100(preset: str, name: str, seconds: int):
    return run_training(preset, name, seconds, "H100")


@app.function(gpu="H200", **FUNCTION_OPTIONS)
def train_h200(preset: str, name: str, seconds: int):
    return run_training(preset, name, seconds, "H200")


@app.function(gpu="H100!:2", **FUNCTION_OPTIONS)
def train_h100_dual(preset: str, name: str, seconds: int):
    return run_training(preset, name, seconds, "H100", gpu_count=2)


@app.function(gpu="B200", **FUNCTION_OPTIONS)
def train_b200(preset: str, name: str, seconds: int):
    return run_training(preset, name, seconds, "B200")


@app.function(gpu="B300", **FUNCTION_OPTIONS)
def train_b300(preset: str, name: str, seconds: int):
    return run_training(preset, name, seconds, "B300")


def download_run(name):
    destination = ROOT / "runs" / name
    destination.mkdir(parents=True, exist_ok=False)
    for entry in volume.iterdir(name, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = PurePosixPath(entry.path.lstrip("/")).relative_to(name)
        if ".." in relative.parts:
            raise ValueError("Unsafe artifact path")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            for chunk in volume.read_file(entry.path):
                file.write(chunk)
    return destination


@app.local_entrypoint()
def main(preset: str = "curriculum", name: str = "", seconds: int = 0, gpu: str = "H100"):
    gpu = gpu.upper()
    if gpu not in GPU_TYPES:
        raise ValueError(f"GPU must be one of {GPU_TYPES}")
    name = name or f"modal-{gpu.lower()}-{preset}-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}"
    validate_run(preset, name, seconds)
    if (ROOT / "runs" / name).exists():
        raise ValueError("Local output already exists; choose a new run name")
    if (preset == "dual") != (gpu == "H100:2"):
        raise ValueError("The dual preset requires H100:2; single-GPU presets require one GPU")
    duration = f"at most {seconds}s training/evaluation" if seconds else "train to configured iterations; no application time limit"
    print(f"Modal: {gpu}, {duration}, no automatic retries. Run: {name}")
    selected = {"H100": train_h100, "H100:2": train_h100_dual,
                "H200": train_h200, "B200": train_b200, "B300": train_b300}[gpu]
    result = selected.remote(preset, name, seconds)
    output = download_run(name)
    print(f"Downloaded artifacts to {output}")
    if result["status"] == "failed":
        raise RuntimeError(f"Cloud training failed. See {output / 'training.log'}")
