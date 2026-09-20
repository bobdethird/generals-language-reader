"""CPU-only W&B mirror of existing Modal training logs; never touches the learner."""
from pathlib import Path
import hashlib
import json
import time
import modal

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = "published-20260919-native4-production-eight-b200-8gpu-train"
PROJECT = "generals-language-reader"
app = modal.App("generals-wandb-metrics")
volume = modal.Volume.from_name("generals-policy-checkpoints")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("wandb==0.30.0", "requests==2.32.5")
         .env({"PYTHONPATH": "/project", "WANDB_MODE": "online"})
         .add_local_file(ROOT / "modal_wandb.py", "/project/modal_wandb.py")
         .add_local_file(ROOT / "reader/wandb_sync.py", "/project/reader/wandb_sync.py"))
OPTIONS = dict(image=image, cpu=0.25, memory=512,
               secrets=[modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])],
               volumes={"/runs": volume.with_mount_options(read_only=True)},
               timeout=86400, max_containers=1, retries=0, include_source=False)


def graphql(query, variables=None):
    import os
    import requests
    response = requests.post("https://api.wandb.ai/graphql", timeout=30,
                             auth=("api", os.environ["WANDB_API_KEY"]),
                             json={"query": query, "variables": variables or {}})
    response.raise_for_status()
    result = response.json()
    if result.get("errors"):
        # Do not include request headers, credentials, or full response bodies.
        raise RuntimeError("W&B GraphQL: " + "; ".join(e["message"] for e in result["errors"]))
    return result["data"]


@app.function(**OPTIONS)
def check():
    account = graphql("query { viewer { username entity } }")["viewer"]
    if not account:
        raise RuntimeError("W&B key has no authenticated account")
    return account


def private_project(entity, project, create=True):
    variables = dict(entity=entity, project=project)
    query = """query($entity: String!, $project: String!) {
      project(entityName: $entity, name: $project) { name access }
    }"""
    current = graphql(query, variables)["project"]
    if current is None and create:
        graphql("""mutation($input: UpsertModelInput!) {
          upsertModel(input: $input) { project { name access } }
        }""", {"input": {"entityName": entity, "name": project, "access": "PRIVATE",
                            "description": "Generals self-play learner metrics, mirrored from Modal"}})
        current = graphql(query, variables)["project"]
    if not current or current["access"] not in ("PRIVATE", "RESTRICTED"):
        raise RuntimeError("W&B project must be private before training metrics are uploaded")
    return current


def run_id_for(source):
    return "modal-" + hashlib.sha256(source.encode()).hexdigest()[:16]


@app.function(**OPTIONS)
def inspect_run(source=DEFAULT_SOURCE, entity="", project=PROJECT):
    import wandb
    from reader.wandb_sync import valid_name
    valid_name(source)
    api = wandb.Api()
    entity = entity or api.default_entity
    privacy = private_project(entity, project, create=False)
    matches = list(api.runs(f"{entity}/{project}", filters={"name": run_id_for(source)}))
    if not matches:
        return {"entity": entity, "project": project, "privacy": privacy, "run": None}
    run = matches[0]
    return {"url": run.url, "state": run.state, "privacy": privacy,
            "summary": dict(run.summary), "last_history_step": run.lastHistoryStep}


@app.function(**OPTIONS)
def sync(source=DEFAULT_SOURCE, entity="", project=PROJECT, poll_seconds=30):
    import wandb
    from reader.wandb_sync import lineage, events, public_config, valid_name
    valid_name(source)
    if poll_seconds < 10:
        raise ValueError("Polling interval must be at least ten seconds")
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("W&B account has no default entity; supply --entity")
    private_project(entity, project)
    volume.reload()
    segments = lineage("/runs", source)
    deadline = float(segments[-1]["record"]["deadline_unix"])
    run_id = run_id_for(source)
    seen = set()
    matches = list(api.runs(f"{entity}/{project}", filters={"name": run_id}))
    if matches:
        seen = {row["sync/event_id"] for row in matches[0].scan_history(keys=["sync/event_id"])}
    settings = wandb.Settings(
        console="off", disable_code=True, disable_git=True, save_code=False,
        x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
        x_save_requirements=False, x_file_stream_transmit_interval=5,
        quiet=True, finish_timeout=90,
    )
    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow",
                     name="Published policy · live 8×B200", job_type="metrics-mirror",
                     config=public_config(segments), settings=settings, dir="/tmp",
                     notes="Scalar mirror of the selected checkpoint lineage. Global batch grows at iteration 241. "
                           "Evaluation is against random play on a changing curriculum, not an Elo rating.")
    run.define_metric("iteration")
    for prefix in ("train", "eval", "performance", "curriculum", "timing", "sync"):
        run.define_metric(prefix + "/*", step_metric="iteration", step_sync=False)
    print(json.dumps({"wandb_url": run.url, "privacy": "PRIVATE", "gpu": None,
                      "source": source, "poll_seconds": poll_seconds,
                      "resumed_events": len(seen)}), flush=True)
    terminal_since = None
    highest_iteration = int(run.summary.get("sync/latest_iteration", 0))
    try:
        while True:
            volume.reload()
            segments = lineage("/runs", source)
            uploaded = 0
            for event in events("/runs", segments):
                highest_iteration = max(highest_iteration, event["iteration"])
                if event["sync/event_id"] in seen:
                    continue
                run.log(event)  # W&B step is an event index; iteration is the chart axis.
                seen.add(event["sync/event_id"])
                uploaded += 1
            status = segments[-1]["record"]["status"]
            run.summary.update({"sync/latest_iteration": highest_iteration,
                                "sync/events": len(seen), "sync/training_status": status,
                                "sync/last_poll_unix": time.time()})
            if uploaded:
                print(json.dumps({"uploaded": uploaded, "latest_iteration": highest_iteration,
                                  "events": len(seen), "training_status": status}), flush=True)
            if status not in ("starting", "running"):
                terminal_since = terminal_since or time.time()
                if time.time() - terminal_since >= 60:
                    break  # Two extra reloads capture the trainer's final volume commit.
            if time.time() > deadline + 600:
                run.summary["sync/stopped_reason"] = "training_deadline_plus_ten_minute_grace"
                break
            time.sleep(poll_seconds)
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    return {"url": run.url, "latest_iteration": highest_iteration, "events": len(seen)}


@app.local_entrypoint()
def main(check_only: bool = False, inspect_only: bool = False,
         source: str = DEFAULT_SOURCE, entity: str = "", project: str = PROJECT):
    if check_only:
        print(json.dumps(check.remote()))
    elif inspect_only:
        print(json.dumps(inspect_run.remote(source, entity, project)))
    else:
        call = sync.spawn(source, entity, project)
        report = {"call_id": call.object_id, "source": source, "project": project}
        output = ROOT / "runs/wandb-sync"
        output.mkdir(parents=True, exist_ok=True)
        (output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        print(json.dumps(call.get()), flush=True)
