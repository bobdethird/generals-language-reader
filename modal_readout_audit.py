"""Audit the frozen-readout reader and combine it with the completed comparison."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT

app = modal.App("generals-readout-fidelity-audit")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image
         .add_local_file(ROOT/"scripts/train_decision_listener.py", "/project/scripts/train_decision_listener.py")
         .add_local_file(ROOT/"scripts/audit_decision_contrasts.py", "/project/scripts/audit_decision_contrasts.py")
         .add_local_file(ROOT/"modal_readout_audit.py", "/project/modal_readout_audit.py"))


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=3600,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def audit():
    import subprocess
    import sys
    volume.reload()
    root = Path("/experiments")
    output = root/"iter2000-decision-fidelity-audit-v3"
    output.mkdir(exist_ok=False)
    reader = root/"iter2000-decision-readout-v2"
    try:
        commands = [
            [sys.executable, "-u", "/project/scripts/audit_decision_contrasts.py", "--data", str(root/"iter2000-spatial-data-v3"),
             "--readers", str(reader), "--output", str(output/"readout-contrasts.json")],
            [sys.executable, "-u", "/project/scripts/train_decision_listener.py", "--data", str(root/"iter2000-spatial-data-v3"),
             "--output", str(root/"iter2000-decision-listener-v1"), "--audit", "--audit-device", "cuda",
             "--audit-output", str(output/"readout-context.json"), "--readers", str(reader)],
        ]
        for command in commands:
            print(json.dumps({"stage": "audit", "command": command}), flush=True)
            subprocess.run(command, cwd="/project", check=True)
        previous = root/"iter2000-decision-fidelity-audit-v1"
        old = json.loads((previous/"summary.json").read_text())
        names = old["readers"] + [reader.name]
        scores, results = {}, {}
        for name in names:
            report = json.loads((root/name/"report.json").read_text())
            record = next(r for r in report["validation_history"] if r["step"] == report["best_step"])
            m = record["metrics"]["real"]
            scores[name] = [m["move_exact_accuracy"] or 0, m["fully_correct_description"], -m["caption_nll"]]
            results[name] = {"best_step": report["best_step"], "test": report["test"],
                             "frozen_head_test": report.get("frozen_head_test"),
                             "player_unchanged": report["frozen_player_unchanged"],
                             "language_unchanged": report["frozen_language_unchanged"]}
        summary = {"selected_reader": max(scores, key=scores.get), "validation_selection_scores": scores,
                   "readers": results,
                   "scope": "Move reporting and semantic-context usefulness; no claim of recovered causal strategic motives."}
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+'\n')
        for destination, older, newer in (("contrasts.json", "contrasts.json", "readout-contrasts.json"),
                                           ("context-audit.json", "context-audit.json", "readout-context.json")):
            a = json.loads((previous/older).read_text())
            b = json.loads((output/newer).read_text())
            if a["positions"] != b["positions"]:
                raise ValueError("Audit population mismatch")
            a["readers"].update(b["readers"])
            (output/destination).write_text(json.dumps(a, indent=2)+'\n')
        return summary
    finally:
        volume.commit()


@app.function(image=image, cpu=.25, memory=2048, timeout=28800, retries=0,
              max_containers=1, include_source=False)
def coordinate():
    for call in ("fc-01M2YT7X2ZPWFHAD8FYVQVH8RK", "fc-01M2YR10V1ZVVV4A6NZPQPQ1Y7"):
        modal.FunctionCall.from_id(call).get()
    return audit.remote()


@app.local_entrypoint()
def main():
    call = coordinate.spawn()
    print(json.dumps({"call_id": call.object_id}), flush=True)
