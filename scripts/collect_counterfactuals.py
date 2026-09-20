"""Probe frozen-player snapshots; store numerical evidence and automatic targets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cuda")
from scripts.collect_activations import Config, build_network, jax, eqx, batch_read
import numpy as np
import jax.numpy as jnp
from reader.counterfactuals import make_interventions, summarize_effects, rationale_caption, SCOPE
from reader.decisions import greedy_actions, decision_facts, decision_caption


def infer_fixed(net, obs, masks, temporal, batch_size=128):
    """Use the same XLA batch shape for baselines, edits, and identity controls."""
    results=[]
    for start in range(0,len(obs),batch_size):
        parts=[x[start:start+batch_size] for x in (obs,masks,temporal)]
        n=len(parts[0])
        padded=[np.concatenate([x,np.repeat(x[-1:],batch_size-n,axis=0)]) if n<batch_size else x for x in parts]
        # Upstream rollout casts observations to BF16 BEFORE normalization.
        # Export stores those values as FP32; feeding FP32 back changes division
        # rounding and can flip actions. Restore the original input dtype.
        padded[0] = jnp.asarray(padded[0],dtype=jnp.bfloat16)
        # Keep the original export function and batch shape for every test.
        all_outputs = batch_read(net,*padded)
        results.append(np.asarray(all_outputs[1])[:n])
    return np.concatenate(results)


def extract(checkpoint, source, output, count=None, seed=38001, batch_size=128):
    started = time.monotonic()
    cfg = Config.from_yaml(ROOT / "configs/averagejoe-published.yaml")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    net = eqx.tree_deserialise_leaves(checkpoint, build_network(cfg, jax.random.PRNGKey(cfg.seed)))
    rows = [json.loads(line) for line in (source / "captions.jsonl").read_text().splitlines()]
    with np.load(source / "samples.npz", allow_pickle=False) as z:
        indices = np.arange(len(rows)) if count is None else np.sort(np.random.default_rng(seed).choice(
            len(rows), size=min(count, len(rows)), replace=False))
        arrays = {key: z[key][indices] for key in ("activations", "observations", "masks", "temporal", "logits")}
    rows = [rows[i] for i in indices]
    output.mkdir(parents=True, exist_ok=False)
    # Replay the selected population in full batches, matching export sizing.
    # Only the intervention construction is chunked into groups of eight.
    baselines=infer_fixed(net,arrays["observations"],arrays["masks"],arrays["temporal"],batch_size)
    evidence, examples = [], []
    max_error = 0.; agreements = 0
    for start in range(0, len(indices), 8):
        end = min(start + 8, len(indices))
        o, m, t = (arrays[k][start:end] for k in ("observations", "masks", "temporal"))
        baseline = baselines[start:end]
        target = arrays["logits"][start:end]
        agreements += int(np.sum(greedy_actions(baseline)==greedy_actions(target)))
        max_error = max(max_error, float(np.abs(baseline-target).max()))
        altered, histories, valid, metadata = zip(*(make_interventions(a, b, int(c))
            for a, b, c in zip(o, t, greedy_actions(baseline))))
        # Every position gets explicit original-input controls in two batch
        # slots, even when every semantic probe is active.
        obs = np.concatenate([o[:,None],np.stack(altered).reshape(len(o),20,38,24,24),o[:,None]],1).reshape(-1,38,24,24)
        temporal = np.concatenate([t[:,None],np.stack(histories).reshape(len(o),20,2,512),t[:,None]],1).reshape(-1,2,512)
        masks = np.repeat(m,22,axis=0)
        measured = infer_fixed(net, obs, masks, temporal, batch_size).reshape(len(o),22,5184)
        for local, (base, full, active, meta) in enumerate(zip(baseline, measured, valid, metadata)):
            result=full[1:-1].reshape(10,2,5184)
            identities=np.concatenate([full[[0,-1]],result[~active].reshape(-1,5184)])
            f, rationale, audit = summarize_effects(base,result,active,identity_logits=identities,
                                                    archived_logits=target[local])
            # A mismatched replay stays in the data as an explicitly inconclusive
            # case. Its move target remains the original archived player's move.
            audit["archived_action"]=int(greedy_actions(target[local]))
            evidence.append(f)
            facts = decision_facts(o[local], audit["archived_action"])
            row = rows[start+local]
            examples.append({**row, "snapshot_index": int(indices[start+local]),
                "source": str(source), "what_target": decision_caption(facts), "decision_facts": facts,
                "why_target": rationale_caption(rationale), "rationale": rationale,
                "interventions": meta, "evidence_audit": audit})
        if start % 128 == 0:
            print(json.dumps({"stage": "probe", "positions": end, "total": len(indices),
                              "seconds": round(time.monotonic()-started, 2)}), flush=True)
    np.savez_compressed(output / "evidence.npz", activations=arrays["activations"], masks=arrays["masks"],
                        evidence=np.stack(evidence), observations=arrays["observations"],
                        temporal=arrays["temporal"], logits=arrays["logits"])
    (output / "examples.jsonl").write_text("".join(json.dumps(r)+'\n' for r in examples))
    report = {"positions": len(examples), "map_ids": sorted({r["map_id"] for r in rows}),
              "player_sha256": digest, "source_samples_sha256": hashlib.sha256((source/"samples.npz").read_bytes()).hexdigest(),
              "probe_source_sha256": hashlib.sha256((ROOT/"reader/counterfactuals.py").read_bytes()).hexdigest(),
              "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "observation_network_dtype": "bfloat16", "observation_storage_dtype": "float32",
              "nontrivial_explanations": sum(r["rationale"]["probe"] >= 0 for r in examples),
              "numerically_unstable_positions":sum(not r["evidence_audit"]["numerically_stable_action"] for r in examples),
              "original_greedy_agreement":agreements/len(examples),
              "original_logit_max_error": max_error, "seconds": time.monotonic()-started, "scope": SCOPE}
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    (output / "report.json").write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({"stage": "probe_complete", **report}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--seed", type=int, default=38001)
    args = p.parse_args()
    extract(args.checkpoint, args.source, args.output, args.count, args.seed)


if __name__ == "__main__": main()
