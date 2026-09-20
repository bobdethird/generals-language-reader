"""Diagnose saved-vs-recomputed BF16 policy outputs without altering weights."""
from pathlib import Path
import modal
import json
ROOT=Path(__file__).resolve().parent
app=modal.App("generals-probe-precision")
volume=modal.Volume.from_name("generals-reader-experiments")
if modal.is_local():
    from modal_policy import image as base
    image=base.add_local_file(ROOT/"modal_probe_precision.py","/project/modal_probe_precision.py")
else:image=modal.Image.debian_slim(python_version="3.12")

@app.function(image=image,gpu="B200",cpu=4,memory=32768,timeout=1800,
              volumes={"/experiments":volume},include_source=False,retries=0,
              max_containers=1,single_use_containers=True,scaledown_window=2)
def diagnose():
    import numpy as np
    import jax.numpy as jnp
    from scripts.collect_activations import Config,build_network,jax,eqx,batch_read
    from scripts.collect_counterfactuals import infer_fixed
    from reader.counterfactuals import canonical_scores,action_margin
    volume.reload();root=Path("/experiments")
    data=root/"iter3800-spatial-data-v1"
    manifest=json.loads((data/"manifest.json").read_text())
    source=data/manifest["train"]["shards"][2]["path"]
    with np.load(source/"samples.npz",allow_pickle=False) as z:
        arrays={k:z[k] for k in ("observations","masks","temporal","logits","activations")}
    cfg=Config.from_yaml("/project/configs/averagejoe-published.yaml")
    net=eqx.tree_deserialise_leaves(root/"iter3800-reader-transfer-v1/L_7d_gae90_ema_3800.eqx",build_network(cfg,jax.random.PRNGKey(cfg.seed)))
    idx=np.sort(np.random.default_rng(380102).choice(len(arrays["logits"]),512,replace=False))
    selected={k:v[idx] for k,v in arrays.items()}
    subset=infer_fixed(net,selected["observations"],selected["masks"],selected["temporal"])
    repeated=infer_fixed(net,selected["observations"],selected["masks"],selected["temporal"])
    contiguous=infer_fixed(net,arrays["observations"],arrays["masks"],arrays["temporal"])[idx]
    target=selected["logits"]
    results={}
    for name,x in (("subset",subset),("repeat",repeated),("contiguous",contiguous)):
        original=canonical_scores(target);actual=canonical_scores(x)
        errors=np.abs(x-target).max(-1)
        results[name]={"max_logit_error":float(errors.max()),"nonzero_positions":int((errors>0).sum()),
                       "greedy_agreement":float(np.mean(original.argmax(-1)==actual.argmax(-1)))}
    errors=np.abs(subset-target).max(-1)
    bad=np.flatnonzero(errors>0)
    examples=[]
    for i in bad[:16]:
        action=int(canonical_scores(target[i]).argmax())
        examples.append({"snapshot_index":int(idx[i]),"max_error":float(errors[i]),
            "archived_action":action,"actual_action":int(canonical_scores(subset[i]).argmax()),
            "archived_margin":float(action_margin(target[i],action)),
            "actual_margin":float(action_margin(subset[i],action)),
            "subset_contiguous_max_error":float(np.abs(subset[i]-contiguous[i]).max())})
    report={"source":str(source),"positions":len(idx),"variants":results,
            "within_process_repeat_max_error":float(np.abs(subset-repeated).max()),
            "subset_contiguous_max_error":float(np.abs(subset-contiguous).max()),"examples":examples}
    (root/"iter3800-probe-precision-v1.json").write_text(json.dumps(report,indent=2)+'\n');volume.commit()
    print(json.dumps(report),flush=True);return report

@app.local_entrypoint()
def main():
    call=diagnose.spawn();print(json.dumps({"call_id":call.object_id}),flush=True)
