"""Collect one reproducible shard group with independent map sections per lane."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_activations import (collect_split, collection_config, Config, GeneralsEnv,
                                         build_network, jax, eqx)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--batches", type=int, required=True)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--pool-size", type=int, default=6272)
    p.add_argument("--seed-base", type=int, default=60000000)
    p.add_argument("--extra-split", choices=["validation", "test"])
    args = p.parse_args()
    if any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("Dataset collection requires the requested NVIDIA GPU")
    cfg = collection_config(Config.from_yaml(ROOT / "configs/averagejoe-published.yaml"), 4, args.pool_size)
    network = eqx.tree_deserialise_leaves(args.checkpoint, build_network(cfg, jax.random.PRNGKey(cfg.seed)))
    env = GeneralsEnv(min_grid_size=cfg.min_grid_size, max_grid_size=cfg.max_grid_size,
                     pad_to=cfg.pad_to, min_generals_distance=cfg.min_generals_distance,
                     max_generals_distance=cfg.max_generals_distance, truncation=cfg.truncation,
                     pool_size=cfg.pool_size, castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
                     num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
                     mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max))
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    record = {"split": args.split, "worker": args.worker, "checkpoint_sha256": checkpoint_hash,
              "config": cfg.to_dict(), "steps": args.steps, "stride": args.stride,
              "shards": [], "source": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       for name in ["scripts/collect_activations.py", "scripts/collect_spatial_dataset.py"]}}
    plans = [(args.split, batch) for batch in range(args.batches)]
    if args.extra_split:
        plans.extend((args.extra_split, batch) for batch in range(2))
    for split, batch in plans:
        split_offset = {"train": 0, "validation": 10000000, "test": 20000000}[split]
        seed = args.seed_base + split_offset + args.worker * 100000 + batch * 1000
        if not 0 <= seed < 2**32:
            raise ValueError("Collection seed outside the PRNG key range")
        name = f"{split}-w{args.worker}-b{batch}"
        result = collect_split(network, cfg, seed=seed, n_envs=args.envs, steps=args.steps,
                               stride=args.stride, output=args.output, split=name, read_batch_size=128,
                               env=env, diverse_pool_starts=True, compact_activations=False)
        result["path"] = name
        result["split"] = split
        record["shards"].append(result)
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"stage": "shard_complete", "split": split, "worker": args.worker,
                          "batch": batch, "samples": result["samples"], "games": result["games"],
                          "unique_maps": result["unique_maps"], "seconds": result["seconds"]}), flush=True)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    record["player_checkpoint_unchanged"] = True
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
