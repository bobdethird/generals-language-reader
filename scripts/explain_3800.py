"""Report a frozen 3800 move and generate a reason from fresh intervention tests.

Both decoders run after the player's action is selected. Raw generated text is
preserved; unsupported text is marked unverified, never silently repaired.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-checkpoint", type=Path, required=True)
    p.add_argument("--what-reader", type=Path, required=True)
    p.add_argument("--why-reader", type=Path, required=True)
    p.add_argument("--evidence-shard", type=Path, required=True,
                   help="A collected shard containing evidence.npz and report.json")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    os.environ["JAX_PLATFORMS"] = args.device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("HF_HOME", str(ROOT/".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import jax.numpy as jnp
    import torch
    from huggingface_hub import snapshot_download
    from scripts.collect_activations import Config, build_network, jax, eqx, batch_read
    from scripts.collect_counterfactuals import infer_fixed
    from reader.counterfactuals import make_interventions, summarize_effects, parse_rationale, verify_rationale, SCOPE
    from reader.decision_factory import restore_reader
    from reader.decision_data import pack_legal_masks
    from reader.decisions import greedy_actions, decode_action, decision_facts, score_decisions
    from reader.why_language import WhyLanguageReader, pack_why_inputs
    sha = hashlib.sha256(args.player_checkpoint.read_bytes()).hexdigest()
    report = json.loads((args.evidence_shard/"report.json").read_text())
    what_checkpoint = torch.load(args.what_reader, weights_only=True, map_location="cpu")
    why_checkpoint = torch.load(args.why_reader, weights_only=True, map_location="cpu")
    if any(x != sha for x in (report["player_sha256"], what_checkpoint["configuration"]["player_sha256"],
                              why_checkpoint["configuration"]["player_sha256"])):
        raise ValueError("Player, both readers, and snapshot must belong to the same checkpoint")
    if what_checkpoint["model"] != why_checkpoint["model"]:
        raise ValueError("Language backbone revisions differ")
    cfg = Config.from_yaml(ROOT/"configs/averagejoe-published.yaml")
    net = eqx.tree_deserialise_leaves(args.player_checkpoint, build_network(cfg,jax.random.PRNGKey(cfg.seed)))
    with np.load(args.evidence_shard/"evidence.npz", allow_pickle=False) as z:
        if not 0 <= args.index < len(z["activations"]): raise ValueError("Snapshot index out of bounds")
        o,m,t = (z[k][args.index:args.index+1] for k in ("observations","masks","temporal"))
    hidden, logits, value, reconstructed, reconstructed_value = map(np.asarray, batch_read(net,jnp.asarray(o,dtype=jnp.bfloat16),m,t))
    np.testing.assert_allclose(logits,reconstructed,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(value,reconstructed_value,atol=2e-5,rtol=2e-5)
    measured_logits = infer_fixed(net,o,m,t)
    # The actor chooses using its original forward output. A differently batched
    # probe may only explain it; it can never replace the actor's selected move.
    action = int(greedy_actions(logits)[0])
    altered,history,active,metadata = make_interventions(o[0],t[0],action)
    probe_obs=np.concatenate([o,altered.reshape(-1,38,24,24),o])
    probe_temporal=np.concatenate([t,history.reshape(-1,2,512),t])
    modified = infer_fixed(net,probe_obs,np.repeat(m,22,0),probe_temporal)
    probe_logits=modified[1:-1].reshape(10,2,5184)
    identity_logits=np.concatenate([modified[[0,-1]],probe_logits[~active].reshape(-1,5184)])
    evidence,rationale,audit = summarize_effects(measured_logits[0],probe_logits,active,
        identity_logits=identity_logits,archived_logits=logits[0])
    what = restore_reader(what_checkpoint,args.device)
    inputs = pack_legal_masks(hidden,m) if what.legal_mask else hidden
    what_text = what.generate(inputs,max_new_tokens=90)[0]
    facts = decision_facts(o[0],action)
    what_metrics,_ = score_decisions([what_text],[facts])
    lock = why_checkpoint["model"]
    path = snapshot_download(lock["model"],revision=lock["revision"],local_files_only=True)
    why = WhyLanguageReader(path,args.device,why_checkpoint.get("architecture","tokenwise"))
    why.adapter.load_state_dict(why_checkpoint["adapter"])
    why.adapter.eval().requires_grad_(False)
    why_text = why.generate(pack_why_inputs(hidden,evidence[None]),max_new_tokens=80)[0]
    why_check=verify_rationale(why_text,audit)
    why_ok=why_check["claims_supported"]
    what_ok = what_metrics["fully_correct_description"] == 1
    row = json.loads((args.evidence_shard/"examples.jsonl").read_text().splitlines()[args.index])
    result = {"player_iteration":3800,"player_sha256":sha,"map_id":row["map_id"],"snapshot_index":args.index,
              "action":decode_action(action),"raw_what":what_text,"raw_why":why_text,
              "what_verified":what_ok,"why_verified":why_ok,
              "verified_what":what_text if what_ok else None,"verified_why":why_text if why_ok else None,
              "verification":what_metrics,"measured_influence":rationale,"probe_measurements":audit,
              "reported_influence":parse_rationale(why_text),
              "why_verification":{**why_check,"matches_canonical_target":parse_rationale(why_text)==rationale},
              "intervention_details":metadata,"scope":SCOPE,
              "player_unchanged":hashlib.sha256(args.player_checkpoint.read_bytes()).hexdigest()==sha,
              "what_reader_sha256":hashlib.sha256(args.what_reader.read_bytes()).hexdigest(),
              "why_reader_sha256":hashlib.sha256(args.why_reader.read_bytes()).hexdigest(),
              "note":"Actor decisions are selected before the readers run. No decoder output drives the actor. "
                     "The why decoder receives freshly measured numerical effects, not a target explanation. "
                     "Unverified raw text is retained as an error; no teacher text is substituted."}
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text)


if __name__ == "__main__": main()
