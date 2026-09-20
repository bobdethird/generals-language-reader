"""Evaluate numbered EMA snapshots with the pinned upstream paired match loop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reader.checkpoint_evaluation import (PROTOCOL, PROTOCOL_ID, checkpoints,
                                          result_path, summarize_counts)
from reader.runtime import import_upstream, install_simulator_compat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("/runs"))
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    import_upstream()
    install_simulator_compat()
    import equinox as eqx
    import jax
    import numpy as np
    from config import Config
    from networks import build_network, get_network_bundle
    from evals.agent import Agent
    from generals.core.env import GeneralsEnv
    from reader.upstream_training import make_cached_match, paired_match

    if jax.device_count() != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Checkpoint evaluation requires exactly one GPU")
    paths = checkpoints(args.training_root, args.source)
    references = [int(value) for value in args.references.split(",")]
    if args.candidate not in paths or any(ref not in paths or ref > args.candidate for ref in references):
        raise ValueError("Evaluation needs confirmed numbered checkpoints")
    config_path = args.training_root / args.source / "requested-config.yaml"
    config_bytes = config_path.read_bytes()
    if config_bytes != (ROOT / "configs/averagejoe-published.yaml").read_bytes():
        raise ValueError("Checkpoint model configuration differs from pinned published recipe")
    cfg = Config.from_yaml(config_path)
    bundle = get_network_bundle(cfg.network)
    skeleton = build_network(cfg, jax.random.PRNGKey(0))
    hashes = {}

    def load(iteration):
        path = paths[iteration]
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        model = eqx.tree_deserialise_leaves(path, skeleton)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError("Checkpoint changed while being read")
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))):
            raise ValueError("Nonfinite checkpoint weights")
        hashes[iteration] = before
        return Agent(model, cfg, bundle, name=f"ema_{iteration}")

    candidate = load(args.candidate)
    settings = {key: value for key, value in PROTOCOL.items() if key in (
        "min_grid_size", "max_grid_size", "pad_to", "min_generals_distance",
        "max_generals_distance", "truncation", "pool_size")}
    settings.update({key: tuple(PROTOCOL[key]) for key in (
        "castle_val_range", "num_cities_range", "mountain_density_range")})
    env = GeneralsEnv(**settings)
    print("Generating fixed evaluation map pool", flush=True)
    pool, _ = env.reset(jax.random.PRNGKey(PROTOCOL["pool_seed"]))
    jax.block_until_ready(pool)
    play = make_cached_match()
    args.output.mkdir(parents=True, exist_ok=True)
    for reference_iteration in references:
        output = result_path(args.output, args.candidate, reference_iteration)
        if output.exists():
            previous = json.loads(output.read_text())
            if previous["protocol_id"] != PROTOCOL_ID:
                raise ValueError("Existing result uses a different protocol")
            continue
        reference = candidate if reference_iteration == args.candidate else load(reference_iteration)
        hashes[reference_iteration] = hashes.get(reference_iteration, hashes[args.candidate])
        started = time.monotonic()
        counts = dict(wins=0, losses=0, draws=0)
        batches = []
        # One full-shape batch validates a symmetric self-match; real comparisons use 512 games.
        is_control = reference_iteration == args.candidate
        games = 2 * PROTOCOL["maps_per_batch"] if is_control else PROTOCOL["games"]
        for batch in range(games // (2 * PROTOCOL["maps_per_batch"])):
            if time.time() >= args.deadline - 30:
                print("Evaluation deadline reached; unfinished pair remains queued", flush=True)
                return
            key = jax.random.fold_in(jax.random.PRNGKey(PROTOCOL["match_seed"]), batch)
            result = paired_match(play, candidate, reference, env, pool,
                                  PROTOCOL["maps_per_batch"], PROTOCOL["truncation"], key)
            batches.append(result)
            counts = {key: counts[key] + result[key] for key in counts}
            print(json.dumps({"candidate": args.candidate, "reference": reference_iteration,
                              "batch": batch + 1, **counts}), flush=True)
        summary = summarize_counts(counts)
        if summary["games"] != games or (is_control and summary["wins"] != summary["losses"]):
            raise AssertionError("Paired match accounting or self-play symmetry failed")
        row = dict(protocol_id=PROTOCOL_ID, protocol=PROTOCOL, source=args.source,
                   candidate_iteration=args.candidate, reference_iteration=reference_iteration,
                   candidate_sha256=hashes[args.candidate], reference_sha256=hashes[reference_iteration],
                   config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                   upstream=json.loads((ROOT / "upstream-lock.json").read_text()),
                   completed_at=datetime.now(timezone.utc).isoformat(),
                   gpu=str(jax.devices()[0]), seconds=time.monotonic() - started,
                   control=is_control, batches=batches, **summary)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(output)
        print("CHECKPOINT_EVALUATION " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
