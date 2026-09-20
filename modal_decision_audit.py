"""Run final independent audits after all three readers and observer fits finish."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT

app = modal.App("generals-decision-fidelity-audit")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image
         .add_local_file(ROOT / "scripts/train_decision_listener.py", "/project/scripts/train_decision_listener.py")
         .add_local_file(ROOT / "scripts/audit_decision_contrasts.py", "/project/scripts/audit_decision_contrasts.py")
         .add_local_file(ROOT / "modal_decision_audit.py", "/project/modal_decision_audit.py"))


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=7200,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def run_audits():
    import subprocess
    import sys
    import torch
    volume.reload()
    root = Path("/experiments")
    output = root/"iter2000-decision-fidelity-audit-v1"
    output.mkdir(exist_ok=False)
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    if "B200" not in hardware or not torch.cuda.is_available():
        raise RuntimeError("Expected B200")
    names = ["iter2000-decision-comparison-v2-spatial", "iter2000-decision-comparison-v2-policy_aux",
             "iter2000-decision-mask-v1-policy_mask"]
    readers = [str(root/name) for name in names]
    commands = [
        [sys.executable, "-u", "/project/scripts/audit_decision_contrasts.py", "--data", str(root/"iter2000-spatial-data-v3"),
         "--readers", *readers, "--output", str(output/"contrasts.json")],
        [sys.executable, "-u", "/project/scripts/train_decision_listener.py", "--data", str(root/"iter2000-spatial-data-v3"),
         "--output", str(root/"iter2000-decision-listener-v1"), "--audit", "--audit-device", "cuda",
         "--audit-output", str(output/"context-audit.json"), "--readers", *readers],
    ]
    try:
        for command in commands:
            print(json.dumps({"stage": "audit_command", "hardware": hardware, "command": command}), flush=True)
            subprocess.run(command, cwd="/project", check=True)
        # Choose the reader using validation alone, never these audit outcomes.
        scores = {}
        for name in names:
            report = json.loads((root/name/"report.json").read_text())
            record = next(r for r in report["validation_history"] if r["step"] == report["best_step"])
            m = record["metrics"]["real"]
            scores[name] = [m["move_exact_accuracy"] or 0, m["fully_correct_description"], -m["caption_nll"]]
        selected = max(scores, key=scores.get)
        summary = {"selected_reader": selected, "validation_selection_scores": scores,
                   "readers": names, "hardware": hardware,
                   "scope": "Decision fidelity and independent descriptive-context audit; strategic causal motives remain unproven."}
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+'\n')
        return summary
    finally:
        volume.commit()


@app.function(image=image, cpu=.25, memory=2048, timeout=28800,
              retries=0, max_containers=1, include_source=False)
def coordinate():
    calls = ["fc-01M2YPZJE67NBJPFHKB2RTERMA", "fc-01M2YQP54AT651NGX85GYRP5NG",
             "fc-01M2YQE4KHTYH8HCPCQBGTD1W8"]
    for call_id in calls:
        modal.FunctionCall.from_id(call_id).get()
    return run_audits.remote()


@app.local_entrypoint()
def main():
    call = coordinate.spawn()
    print(json.dumps({"coordinator_call_id": call.object_id}), flush=True)
