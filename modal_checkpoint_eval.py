"""CPU checkpoint watcher and on-demand H100 evaluation; training stays untouched."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import modal
from modal_policy import image as base_image, volume as training_volume, ROOT
from reader.checkpoint_evaluation import (PROTOCOL, PROTOCOL_ID, MAX_REFERENCES, checkpoints,
                                          pending_pairs, read_results, validate_source,
                                          wandb_metrics)

DEFAULT_SOURCE = "published-20260919-native4-production-eight-b200-8gpu-train"
PROJECT = "generals-language-reader"
app = modal.App("generals-checkpoint-evaluation")
results_volume = modal.Volume.from_name("generals-checkpoint-evaluations", create_if_missing=True)
image = base_image.add_local_file(ROOT / "modal_checkpoint_eval.py", "/project/modal_checkpoint_eval.py")
COMMON = dict(image=image, volumes={"/runs": training_volume.with_mount_options(read_only=True),
                                    "/evaluations": results_volume},
              timeout=86400, retries=0, max_containers=1, include_source=False)


def run_id(source):
    return f"h2h-{hashlib.sha256(source.encode()).hexdigest()[:12]}-{PROTOCOL_ID}"


def output_directory(source):
    return Path("/evaluations") / validate_source(source) / PROTOCOL_ID


@app.function(gpu="H100!", cpu=(4, 4), memory=(16384, 32768),
              startup_timeout=300, scaledown_window=2, single_use_containers=True, **COMMON)
def evaluate(source, candidate, references, deadline, tracking_source=""):
    """Allocate a GPU only for a concrete candidate; persist completed pairs."""
    import subprocess
    from reader.process_run import run_bounded
    validate_source(source)
    if time.time() >= deadline - 30:
        return {"status": "deadline"}
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "H100" not in hardware or len(hardware.splitlines()) != 1:
        raise RuntimeError(f"Expected one H100; got {hardware}")
    print("EVALUATOR HARDWARE " + hardware, flush=True)
    training_volume.reload()
    results_volume.reload()
    output = output_directory(tracking_source or source)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["JAX_COMPILATION_CACHE_DIR"] = f"/evaluations/compiler-cache-h100-{PROTOCOL_ID}"
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    command = [sys.executable, "-u", "/project/scripts/evaluate_checkpoints.py",
               "--source", source, "--candidate", str(candidate),
               "--references", ",".join(map(str, references)),
               "--output", str(output), "--deadline", str(deadline)]
    try:
        result = run_bounded(command, cwd="/tmp", env=env,
                             log_path=output / f"candidate-{candidate}.log",
                             seconds=max(0.1, deadline - time.time() - 20))
        result["status"] = "deadline" if result["timed_out"] else "completed" if result["exit_code"] == 0 else "failed"
        (output / f"candidate-{candidate}-process.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        results_volume.commit()


def private_project(api, entity):
    # The existing metrics project must already be private; never expand access.
    query = """query($entity: String!, $project: String!) {
      project(entityName: $entity, name: $project) { name access }
    }"""
    result = api._service_api.execute_graphql(query, {"entity": entity, "project": PROJECT})
    if isinstance(result, str):
        result = json.loads(result)
    project = result.get("project")
    if not project or project["access"] not in ("PRIVATE", "RESTRICTED"):
        raise RuntimeError("Checkpoint results require the existing private W&B project")


@app.function(cpu=0.25, memory=1024,
              secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])], **COMMON)
def watch(source=DEFAULT_SOURCE, entity="bobdethird", tracking_source=""):
    """Evaluate 100-step snapshots against the three most recent earlier 500-step anchors."""
    import wandb
    os.environ["WANDB_MODE"] = "online"
    validate_source(source)
    tracking_source = validate_source(tracking_source or source)
    api = wandb.Api()
    entity = entity or api.default_entity
    private_project(api, entity)
    training_volume.reload()
    record = json.loads((Path("/runs") / source / "run.json").read_text())
    deadline = float(record["deadline_unix"])
    output = output_directory(tracking_source)
    output.mkdir(parents=True, exist_ok=True)
    identifier = run_id(tracking_source)
    existing = list(api.runs(f"{entity}/{PROJECT}", filters={"name": identifier}))
    logged = {r["match_id"] for r in existing[0].scan_history(keys=["match_id"])} if existing else set()
    run = wandb.init(entity=entity, project=PROJECT, id=identifier, resume="allow",
        name="EMA checkpoint comparisons", job_type="checkpoint-evaluation", dir="/tmp",
        config=None if existing else {"source_run": tracking_source, "protocol": PROTOCOL, "protocol_id": PROTOCOL_ID,
                "gpu": "H100", "deadline_unix": deadline,
                "control": "500 vs 500 validates symmetry; it is not evidence of improvement"},
        settings=wandb.Settings(mode="online", console="off", disable_code=True,
            disable_git=True, save_code=False, x_disable_stats=True, x_disable_meta=True,
            x_disable_machine_info=True, x_save_requirements=False,
            x_file_stream_transmit_interval=5, quiet=True, finish_timeout=60))
    run.config.update({"active_source_run": source, "deadline_unix": deadline,
                       "max_references_per_candidate": MAX_REFERENCES,
                       "schedule": "Every 100 iterations against the three most recent earlier 500-step EMA checkpoints"},
                      allow_val_change=True)
    run.define_metric("iteration")
    run.define_metric("checkpoint/*", step_metric="iteration", step_sync=False)
    run.define_metric("control/*", step_metric="iteration", step_sync=False)
    print(json.dumps({"wandb_url": run.url, "deadline_unix": deadline,
                      "reference_schedule": "Most recent three earlier checkpoints at 500-step intervals",
                      "max_references_per_candidate": MAX_REFERENCES, "gpu": "H100 on demand"}), flush=True)
    retries = {}
    state = {"source": source, "tracking_source": tracking_source,
             "protocol_id": PROTOCOL_ID, "wandb_url": run.url,
             "max_references_per_candidate": MAX_REFERENCES}

    def save_state(**updates):
        state.update(updates, updated_unix=time.time())
        (output / "watcher.json").write_text(json.dumps(state, indent=2) + "\n")
        results_volume.commit()
        run.summary.update({f"watcher/{key}": value for key, value in updates.items()})

    try:
        while True:
            training_volume.reload()
            results_volume.reload()
            available = checkpoints("/runs", source)
            results = read_results(output)
            for row in results:
                event = wandb_metrics(row)
                if event["match_id"] not in logged:
                    run.log(event)
                    logged.add(event["match_id"])
                    print(json.dumps({"logged": event["match_id"], "win_rate": row["win_rate"],
                                      "draw_rate": row["draw_rate"], "control": row["control"]}), flush=True)
            done = {(r["candidate_iteration"], r["reference_iteration"]) for r in results}
            pending = pending_pairs(available, done)
            # Run one symmetric control immediately to validate the actual full model on H100.
            if 500 in available and (500, 500) not in done:
                pending.insert(0, (500, 500))
            save_state(status="waiting", available_checkpoints=list(available),
                       completed_pairs=len(done), pending_pairs=[list(p) for p in pending])
            record = json.loads((Path("/runs") / source / "run.json").read_text())
            if time.time() >= deadline - 120:
                save_state(status="deadline", training_status=record["status"])
                break
            if pending:
                candidate = pending[0][0]
                references = [ref for c, ref in pending if c == candidate]
                save_state(status="evaluating", candidate=candidate, references=references)
                job = evaluate.remote(source, candidate, references, deadline, tracking_source)
                if job["status"] == "failed":
                    retries[candidate] = retries.get(candidate, 0) + 1
                    if retries[candidate] >= 3:
                        raise RuntimeError(f"Evaluation failed three times for checkpoint {candidate}; see saved logs")
                    time.sleep(30)
                continue
            if record["status"] not in ("starting", "running"):
                save_state(status="completed", training_status=record["status"])
                break
            time.sleep(30)
        run.finish()
    except BaseException:
        save_state(status="failed")
        run.finish(exit_code=1)
        raise
    return state


@app.local_entrypoint()
def main(source: str = DEFAULT_SOURCE, entity: str = "bobdethird", tracking_source: str = ""):
    call = watch.spawn(source, entity, tracking_source)
    local = ROOT / "runs/checkpoint-evaluation"
    local.mkdir(parents=True, exist_ok=True)
    report = {"source": source, "tracking_source": tracking_source or source,
              "call_id": call.object_id, "protocol_id": PROTOCOL_ID,
              "wandb_url": f"https://wandb.ai/{entity}/{PROJECT}/runs/{run_id(tracking_source or source)}"}
    (local / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    print(json.dumps(call.get()), flush=True)
