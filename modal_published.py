"""Benchmark the published model, then train within the user's campaign deadline."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import shutil
import subprocess
import sys
import tarfile
import time

import modal
from modal_policy import image as base_image, volume, ROOT

app = modal.App("generals-published-training")
TRAINING_TARGET = 30000  # User's target; benchmark iterations are short local windows.
image = base_image.add_local_file(ROOT / "modal_published.py", "/project/modal_published.py")
OPTIONS = dict(image=image, cpu=(8, 8), memory=(32768, 65536), volumes={"/runs": volume},
               timeout=86400, startup_timeout=300, retries=0, min_containers=0,
               max_containers=1, scaledown_window=2, single_use_containers=True,
               include_source=False)


def copy_compilation_cache(source, target):
    archive = source.parent / 'compiler-cache.tar.gz'
    if archive.exists():
        local_archive = Path('/tmp') / (source.parent.name + '-compiler-cache.tar.gz')
        shutil.copyfile(archive, local_archive)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_archive, 'r:gz') as file:
            file.extractall(target, filter='data')
        return
    # Executable caches are compact. XLA's nested per-fusion autotune directory
    # can contain thousands of tiny files and stalls on remote-volume metadata.
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("xla_gpu_per_fusion_autotune_cache_dir"))


def save_compilation_cache(source, output):
    # Pack locally: thousands of individual remote-volume writes are expensive.
    local_archive = Path('/tmp') / (source.name + '.tar.gz')
    with tarfile.open(local_archive, 'w:gz', compresslevel=1) as archive:
        archive.add(source, arcname='.')
    target = output / 'compiler-cache.tar.gz'
    temporary = output / 'compiler-cache.tar.gz.tmp'
    shutil.copyfile(local_archive, temporary)
    temporary.replace(target)


def run_published(gpu, count, name, iterations, deadline, cache_from="", resume_from="", batch_mode="native"):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
        raise ValueError("Invalid run name")
    deadline = float(deadline)
    if time.time() >= deadline - 120:
        return dict(status="deadline", name=name)
    output = Path("/runs") / name
    output.mkdir(exist_ok=False)
    cache = Path(f"/tmp/jax-cache-{name}")
    from reader.published_training import batch_settings
    batches = batch_settings(count, batch_mode)
    is_training = iterations == TRAINING_TARGET
    record = dict(name=name, gpu=gpu, gpu_count=count, iterations=iterations,
                  target_iterations=TRAINING_TARGET,
                  started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  deadline_unix=deadline, status="starting", **batches,
                  reference_evaluation="Unavailable: authors' checkpoint bank is not published",
                  mode="training" if is_training else "benchmark")
    started = time.time()
    try:
        import jax
        if jax.device_count() != count or any(d.platform != "gpu" for d in jax.devices()):
            raise RuntimeError(f"Expected {count} GPUs; got {jax.devices()}")
        record["hardware"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
        if any(gpu not in line for line in record["hardware"].splitlines()):
            raise RuntimeError(f"Unexpected hardware: {record['hardware']}")
        print(f"PUBLISHED HARDWARE {record['hardware']}", flush=True)
        source = Path("/project/configs/averagejoe-published.yaml")
        upstream = Path("/project/vendor/averagejoe/configs/custom/L_7d_gae90.yaml")
        if source.read_bytes() != upstream.read_bytes():
            raise ValueError("Published config differs from pinned upstream")
        config = output / "requested-config.yaml"
        config.write_bytes(source.read_bytes())
        record["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        record["upstream"] = json.loads(Path("/project/upstream-lock.json").read_text())
        record["support_sha256"] = hashlib.sha256(Path("/project/reader/published_training.py").read_bytes()).hexdigest()
        if cache_from:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", cache_from):
                raise ValueError("Invalid cache source")
            previous = Path("/runs") / cache_from
            source_record = json.loads((previous / "run.json").read_text())
            if (source_record["gpu"], source_record["config_sha256"]) != (gpu, record["config_sha256"]):
                raise ValueError("Compilation cache must match GPU type and recipe")
            copy_compilation_cache(previous / "compiler-cache", cache)
            record["compilation_cache_source"] = cache_from
            record["compilation_cache_source_gpu_count"] = source_record["gpu_count"]
            print(f"Reusing {len(list(cache.iterdir()))} compilation cache files from {cache_from}", flush=True)
        # Keep the recipe intact and make the missing external evaluation asset explicit.
        references = output / "checkpoints/references/L_new"
        references.mkdir(parents=True)
        (references / "h2h.json").write_text('{"h2h": {}}\n')
        effective_config = config
        stage_offset = 0
        if resume_from:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", resume_from):
                raise ValueError('Invalid resume source')
            previous = Path('/runs')/resume_from
            previous_record = json.loads((previous/'run.json').read_text())
            if previous_record['config_sha256'] != record['config_sha256']:
                raise ValueError('Resume recipe does not match')
            progress = json.loads((previous/'progress.json').read_text())
            from ruamel.yaml import YAML
            from reader.published_training import resume_settings
            yaml = YAML(typ='safe')
            target = TRAINING_TARGET if is_training else progress["iteration"] + iterations
            resolved = resume_settings(yaml.load(config.read_text()),progress,target,previous)
            stage_offset = progress['stage']
            effective_config = output/'resume-config.yaml'
            with effective_config.open('w') as file:
                yaml.dump(resolved,file)
            checkpoints = {field: hashlib.sha256(Path(resolved[field]).read_bytes()).hexdigest()
                           for field in ('init_checkpoint','ema_checkpoint')}
            record['resume'] = dict(source=resume_from, **progress, sha256=checkpoints,
                                   environment_state='Fresh games on the new workers; learner and curriculum state preserved')
            print(f"Resuming published learner at iteration {progress['iteration']}, stage {stage_offset}",flush=True)
        command = [sys.executable, "-u", "/project/scripts/policy_entry.py", "--config", str(effective_config)]
        overrides = {}
        if is_training:
            # No author reference weights were published. An empty bank already
            # skips every reference match, but upstream still builds an unused
            # pool. Avoid that setup on production resumes; preserve the already
            # launched benchmark configuration for the hardware comparison.
            overrides["ref_eval_every"] = 0
            record["reference_evaluation"] = "Disabled: authors' checkpoint bank is not published; unused pool setup skipped"
        if not resume_from or not is_training:
            overrides["num_iters"] = iterations
        if batches["per_device_envs"] != 512:
            overrides.update(num_envs=batches["per_device_envs"], minibatch_size=batches["per_device_minibatch"])
        for key, value in overrides.items():
            command.extend(["--" + key, str(value)])
        record["runtime_overrides"] = overrides
        record["status"] = "running"
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        volume.commit()
        environment = dict(os.environ, AVERAGEJOE_PUBLISHED_SUPPORT="1",
                           AVERAGEJOE_DEADLINE=str(deadline),
                           AVERAGEJOE_BATCH_MODE=batch_mode,
                           AVERAGEJOE_STAGE_OFFSET=str(stage_offset),
                           PYTHONFAULTHANDLER="1",
                           JAX_COMPILATION_CACHE_DIR=str(cache))
        from reader.process_run import run_bounded
        record.update(run_bounded(command, cwd=output, env=environment,
                                  log_path=output / "training.log", seconds=max(1, deadline-time.time()-30)))
        progress_file = output / "progress.json"
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        record["progress"] = progress
        record["status"] = ("deadline" if record["timed_out"] or progress.get("stop_reason") == "deadline"
                            else "checkpointed" if progress.get("stop_reason") == "handoff"
                            else "completed" if record["exit_code"] == 0 else "failed")
        timing = output / "iteration-timing.jsonl"
        if timing.exists():
            rows = [json.loads(line) for line in timing.read_text().splitlines()]
            warm = rows[2:]
            if warm:
                # Interval between completed updates includes eval, EMA, pool resets and logging.
                intervals = [b["timestamp"]-a["timestamp"] for a, b in zip(rows[1:-1], rows[2:])]
                record["benchmark"] = dict(measured_iterations=len(warm),
                    mean_update_seconds=statistics.mean(r["update_seconds"] for r in warm),
                    mean_iteration_wall_seconds=statistics.mean(intervals),
                    median_iteration_wall_seconds=statistics.median(intervals),
                    mean_rollout_seconds=statistics.mean(r["rollout_seconds"] for r in warm),
                    mean_ppo_seconds=statistics.mean(r["ppo_seconds"] for r in warm),
                    warm_player_samples_per_second=statistics.mean(r["samples"] for r in warm)/statistics.mean(intervals))
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(record["error"], flush=True)
    finally:
        if cache.exists():
            save_compilation_cache(cache, output)
        record["wall_seconds"] = time.time()-started
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        volume.commit()
    print("PUBLISHED_RESULT " + json.dumps(record), flush=True)
    return record


@app.function(gpu="H100!", **OPTIONS)
def published_h100_1(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H100", 1, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H100!:2", **OPTIONS)
def published_h100_2(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H100", 2, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H100!:4", **OPTIONS)
def published_h100_4(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H100", 4, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H100!:8", **OPTIONS)
def published_h100_8(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H100", 8, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H200", **OPTIONS)
def published_h200_1(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H200", 1, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H200:2", **OPTIONS)
def published_h200_2(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H200", 2, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H200:4", **OPTIONS)
def published_h200_4(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H200", 4, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="H200:8", **OPTIONS)
def published_h200_8(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("H200", 8, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B200", **OPTIONS)
def published_b200_1(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B200", 1, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B200:2", **OPTIONS)
def published_b200_2(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B200", 2, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B200:4", **OPTIONS)
def published_b200_4(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B200", 4, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B200:8", **OPTIONS)
def published_b200_8(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B200", 8, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B300", **OPTIONS)
def published_b300_1(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B300", 1, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B300:2", **OPTIONS)
def published_b300_2(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B300", 2, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B300:4", **OPTIONS)
def published_b300_4(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B300", 4, name, iterations, deadline, cache_from, resume_from, batch_mode)


@app.function(gpu="B300:8", **OPTIONS)
def published_b300_8(name: str, iterations: int, deadline: float, cache_from: str = "", resume_from: str = "", batch_mode: str = "native"):
    return run_published("B300", 8, name, iterations, deadline, cache_from, resume_from, batch_mode)


FUNCTIONS = {
    ("H100", 1): published_h100_1,
    ("H100", 2): published_h100_2,
    ("H100", 4): published_h100_4,
    ("H100", 8): published_h100_8,
    ("H200", 1): published_h200_1,
    ("H200", 2): published_h200_2,
    ("H200", 4): published_h200_4,
    ("H200", 8): published_h200_8,
    ("B200", 1): published_b200_1,
    ("B200", 2): published_b200_2,
    ("B200", 4): published_b200_4,
    ("B200", 8): published_b200_8,
    ("B300", 1): published_b300_1,
    ("B300", 2): published_b300_2,
    ("B300", 4): published_b300_4,
    ("B300", 8): published_b300_8,
}


@app.local_entrypoint()
def main(gpus: str = "B200:4", iterations: int = TRAINING_TARGET,
         campaign: str = "published-20260919", cache_from: str = "", resume_from: str = "", deadline: str = "2026-09-20T04:19:03+00:00", batch_mode: str = "native4"):
    deadline_unix = dt.datetime.fromisoformat(deadline).timestamp()
    selections = [(gpu, int(count)) for gpu, count in (s.split(":") for s in gpus.split(","))]
    jobs = []
    for gpu, count in selections:
        name = f"{campaign}-{gpu.lower()}-{count}gpu-{'train' if iterations == TRAINING_TARGET else 'bench'}"
        call = FUNCTIONS[(gpu, count)].spawn(name, iterations, deadline_unix, cache_from, resume_from, batch_mode)
        jobs.append((name, call))
        print(f"LAUNCHED {name}: {call.object_id}", flush=True)
    local = ROOT / "runs" / campaign
    local.mkdir(parents=True, exist_ok=True)
    (local / "launches.json").write_text(json.dumps(dict(deadline=deadline, batch_mode=batch_mode, jobs=[
        dict(name=name, call_id=call.object_id) for name, call in jobs]), indent=2) + "\n")
    # Calls run concurrently; collect each result without downloading large model files.
    for name, call in jobs:
        result = call.get()
        (local / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        for filename in ("training.log", "metrics.jsonl", "iteration-timing.jsonl", "progress.json"):
            try:
                with (local / f"{name}-{filename}").open("wb") as file:
                    for chunk in volume.read_file(f"{name}/{filename}"):
                        file.write(chunk)
            except FileNotFoundError:
                pass
        print("RESULT " + json.dumps(result), flush=True)
