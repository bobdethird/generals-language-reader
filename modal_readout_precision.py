"""Diagnose frozen policy-head arithmetic on training data only."""
from pathlib import Path
import json
import modal
from modal_spatial_training import image as base_image, ROOT
app = modal.App("generals-readout-precision")
volume = modal.Volume.from_name("generals-reader-experiments", create_if_missing=True)
image = (base_image.add_local_file(ROOT/"runs/iter2000-policy-head-full.pt", "/project/head-full.pt")
         .add_local_file(ROOT/"modal_readout_precision.py", "/project/modal_readout_precision.py"))


@app.function(image=image, gpu="B200", cpu=8, memory=32768, timeout=1800,
              volumes={"/experiments": volume}, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def diagnose():
    import numpy as np
    import torch
    volume.reload()
    torch.set_num_threads(8)
    root = Path("/experiments/iter2000-spatial-data-v3")
    manifest = json.loads((root/"manifest.json").read_text())
    shard = root/manifest["train"]["shards"][0]["path"]
    with np.load(shard/"samples.npz", allow_pickle=False) as a:
        h = torch.from_numpy(a["activations"].copy())[:, 3:]
        teacher = torch.from_numpy(a["logits"].copy())
    head = torch.load("/project/head-full.pt", map_location="cuda", weights_only=True)
    assert head["player_sha256"] == manifest["checkpoint_sha256"]
    report = {"positions": len(h), "max_hidden_bf16_rounding": float((h-h.bfloat16().float()).abs().max()), "modes": {}}
    modes = ("bf16_dot_float_bias", "bf16_dot_bf16_bias", "float_dot_float_bias", "float_dot_round_bias",
             "raw_float_head", "raw_input_rounded_head", "bf16_fused")
    for reduced in (True, False):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
        torch.set_float32_matmul_precision("highest")
        for mode in modes:
            correct = 0
            max_error = abs_sum = 0.
            count = 0
            for start in range(0, len(h), 128):
                x, target = h[start:start+128].cuda(), teacher[start:start+128].cuda()
                w, b = head["weight"], head["bias"]
                if mode == "bf16_dot_float_bias": z=(x.bfloat16()@w.bfloat16().T).float()+b
                elif mode == "bf16_dot_bf16_bias": z=((x.bfloat16()@w.bfloat16().T)+b.bfloat16()).float()
                elif mode == "float_dot_float_bias": z=x.bfloat16().float()@w.T+b
                elif mode == "float_dot_round_bias": z=(x.bfloat16().float()@w.T).bfloat16().float()+b
                elif mode == "raw_float_head": z=x@head["raw_weight"].T+head["raw_bias"]
                elif mode == "raw_input_rounded_head": z=x@w.T+b
                else:z=torch.nn.functional.linear(x.bfloat16(),w.bfloat16(),b.bfloat16()).float()
                z=z.reshape(-1,8,8,9,3,3).permute(0,3,1,4,2,5).reshape(-1,5184)
                legal=target>-1e8
                z=z.masked_fill(~legal,-1e9)
                error=(z-target).abs()
                max_error=max(max_error,float(error.max()));abs_sum+=float(error[legal].sum());count+=int(legal.sum())
                correct+=int((z.argmax(-1).clamp_max(4608)==target.argmax(-1).clamp_max(4608)).sum())
            score={"greedy_agreement":correct/len(h),"max_logit_error":max_error,"mean_legal_logit_error":abs_sum/count}
            name=f"{mode}:reduced={reduced}"
            report["modes"][name]=score
            print(json.dumps({"mode":name,**score}),flush=True)
    out=Path("/experiments/iter2000-readout-precision-v1.json")
    out.write_text(json.dumps(report,indent=2)+'\n');volume.commit()
    return report


@app.local_entrypoint()
def main():
    call=diagnose.spawn();print(json.dumps({"call_id":call.object_id}),flush=True)
