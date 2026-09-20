"""Export independent train/validation self-play sets from a frozen checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(ROOT / ".cache/reader-collection"))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from reader.runtime import import_upstream, install_simulator_compat
import_upstream()
install_simulator_compat()
from config import Config
from generals.core.env import GeneralsEnv
from networks import build_network, get_network_bundle
from train.rollout_selfplay import collect_rollout
from train.rewards import win_lose_reward
from reader.activations import read_activations, outputs_from_activations
from reader.captions import observation_facts


def episode_ids(terminated, truncated, n_envs):
    """Keep both seats and adjacent frames of an episode in one group."""
    terminated, truncated = np.asarray(terminated), np.asarray(truncated)
    if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
        raise ValueError("Episode boundaries must be boolean, not winner IDs")
    if terminated.shape != truncated.shape or terminated.shape[1] != 2 * n_envs:
        raise ValueError("Unexpected paired rollout dimensions")
    done = (terminated | truncated)[:, :n_envs]
    return np.cumsum(np.concatenate([np.zeros((1, n_envs), dtype=int), done[:-1]]), axis=0)


def collection_config(cfg, curriculum_stage=None, pool_size=None):
    if cfg.curriculum_stages:
        if curriculum_stage is None or not 0 <= curriculum_stage < len(cfg.curriculum_stages):
            raise ValueError("Curriculum configurations require an explicit valid --curriculum-stage")
        stage = cfg.curriculum_stages[curriculum_stage]
        fields = ("min_generals_distance", "max_generals_distance", "castle_val_min",
                  "castle_val_max", "num_cities_min", "num_cities_max")
        cfg = replace(cfg, **{k: getattr(stage, k) for k in fields if getattr(stage, k) is not None})
    elif curriculum_stage is not None:
        raise ValueError("This configuration has no curriculum")
    if pool_size is not None:
        if pool_size < (cfg.max_grid_size - cfg.min_grid_size + 1) ** 2:
            raise ValueError("Pool must contain at least one map of every size combination")
        cfg = replace(cfg, pool_size=pool_size)
    return cfg


def pool_start_indices(pool_size, n_envs):
    """Reserve a separate consecutive reset section for each game lane."""
    width = pool_size // n_envs
    if width < 2:
        raise ValueError("Need at least two pool entries per game lane")
    return np.arange(n_envs, dtype=np.int32) * width, width


@eqx.filter_jit
def batch_read(net, o, m, t):
    h = jax.vmap(read_activations, in_axes=(None, 0, 0))(net, o, t)
    original_logits, original_values, _ = jax.vmap(net._forward)(o, m, t)
    read_logits, read_values = jax.vmap(outputs_from_activations, in_axes=(None, 0, 0))(net, h, m)
    return h, original_logits, original_values, read_logits, read_values


def collect_split(network, cfg, *, seed, n_envs, steps, stride, output, split, read_batch_size=16,
                  env=None, diverse_pool_starts=False, compact_activations=False):
    # Each split has independent map-generation and reset-pool seeds. Both seats
    # and every frame of a game stay together, including auto-reset episodes.
    env = env or GeneralsEnv(min_grid_size=cfg.min_grid_size, max_grid_size=cfg.max_grid_size,
                      pad_to=cfg.pad_to, min_generals_distance=cfg.min_generals_distance,
                      max_generals_distance=cfg.max_generals_distance,
                      truncation=cfg.truncation, pool_size=cfg.pool_size,
                      castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
                      num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
                      mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max))
    started = time.monotonic()
    print(f"{split}: preparing {n_envs} games, {steps} rollout steps", flush=True)
    key, pool_key, state_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    pool, _ = env.reset(pool_key)
    if diverse_pool_starts:
        starts, section_width = pool_start_indices(len(pool.time), n_envs)
        states = jax.tree.map(lambda x: x[jnp.asarray(starts)], pool)
        states = states._replace(pool_idx=jnp.asarray(starts + 1))
    else:
        states = jax.vmap(env.init_state)(jax.random.split(state_key, n_envs))
    bundle = get_network_bundle(cfg.network)
    initial = bundle["init_obs_state"](cfg.pad_to, cfg.pad_to)
    history = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_envs, *x.shape)), initial)
    print(f"{split}: collecting frozen-player self-play", flush=True)
    _, rollout, _, _, _ = collect_rollout(
        states, env, network, key, steps, history, history, cfg.pad_to,
        win_lose_reward, bundle["augment_obs"], bundle["reset_obs_state"], cfg.gamma, pool)
    # The returned tuple includes next_vals at index 6, followed by rewards;
    # terminated and truncated are at indices 8 and 9.
    obs, masks, temporal, actions, _, _, _, _, terminated, truncated, _, _ = rollout
    episode = episode_ids(terminated, truncated, n_envs)
    if diverse_pool_starts and episode.max() >= section_width:
        raise RuntimeError("A lane exhausted its reserved map section; enlarge the collection pool")
    frames = np.arange(0, steps, stride)
    flatten = lambda x: x[frames].reshape((-1, *x.shape[2:]))
    obs, masks, temporal, actions = map(flatten, (obs, masks, temporal, actions))

    del rollout
    chunks = [[], [], []]
    max_error = 0.0
    print(f"{split}: exporting {len(obs)} activation snapshots in batches of {read_batch_size}", flush=True)
    for start in range(0, len(obs), read_batch_size):
        end = start + read_batch_size
        h, logits, values, read_logits, read_values = batch_read(
            network, obs[start:end], masks[start:end], temporal[start:end])
        h, logits, values, read_logits, read_values = map(np.asarray, (h, logits, values, read_logits, read_values))
        np.testing.assert_allclose(read_logits, logits, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(read_values, values, atol=2e-5, rtol=2e-5)
        max_error = max(max_error, float(np.max(np.abs(read_logits - logits))))
        for destination, data in zip(chunks, (h, logits, values)):
            destination.append(data)
    activations, logits, values = [np.concatenate(parts) for parts in chunks]
    if compact_activations:
        compact = activations.astype(np.float16)
        # Normalization may promote bf16 inputs to fp32; never silently quantize.
        np.testing.assert_array_equal(compact.astype(np.float32), activations)
        activations = compact
    host_obs = np.asarray(obs, dtype=np.float32)
    prefix = output / split
    prefix.mkdir()
    np.savez_compressed(prefix / "samples.npz", activations=np.asarray(activations),
                        observations=host_obs, masks=np.asarray(masks),
                        temporal=np.asarray(temporal), actions=np.asarray(actions),
                        logits=np.asarray(logits), values=np.asarray(values))
    map_hashes = {}
    if diverse_pool_starts:
        used = np.unique(starts[None, :] + episode[frames])
        geometry = {name: np.asarray(getattr(pool, name)[used]) for name in
                    ("armies", "ownership", "generals", "castles", "mountains")}
        for i, pool_index in enumerate(used):
            hasher = hashlib.sha256()
            for name, arrays in geometry.items():
                hasher.update(name.encode())
                hasher.update(arrays[i].tobytes())
            map_hashes[int(pool_index)] = hasher.hexdigest()
    rows = []
    for index, o in enumerate(host_obs):
        frame, seat_index = divmod(index, 2 * n_envs)
        seat, lane = divmod(seat_index, n_envs)
        turn = int(frames[frame])
        rows.append({"index": index, "game_id": f"{split}:{seed}:{lane}:{episode[turn, lane]}",
                     "seat": seat, "turn": int(o[14, 0, 0]),
                     "caption": observation_facts(o), "caption_type": "observable_fact_warm_start"})
        if diverse_pool_starts:
            rows[-1]["map_id"] = map_hashes[int(starts[lane] + episode[turn, lane])]
    (prefix / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"{split}: {len(rows)} snapshots; original/readout outputs agree", flush=True)
    return {"samples": len(rows), "map_seed": seed, "environments": n_envs,
            "games": len({row["game_id"] for row in rows}), "seconds": time.monotonic() - started,
            "activation_shape": list(activations.shape[1:]), "head_readout_max_error": max_error,
            "activation_dtype": str(activations.dtype), "diverse_pool_starts": diverse_pool_starts,
            "unique_maps": len(set(map_hashes.values())) if diverse_pool_starts else None,
            "map_ids": sorted(set(map_hashes.values())),
            "unique_general_cells": len(set(map(tuple, np.argwhere((host_obs[:, 6] > 0) &
                                                                   (host_obs[:, 10] > 0))[:, 1:])))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Network-only EMA checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1044)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--only-split", choices=["train", "validation", "test"],
                        help="Export just one split, for a fresh final audit")
    parser.add_argument("--curriculum-stage", type=int, help="Explicit difficulty for curriculum-trained players")
    parser.add_argument("--pool-size", type=int, help="Map pool for this export, independent of training")
    parser.add_argument("--read-batch-size", type=int, default=16)
    args = parser.parse_args()
    if min(args.steps, args.stride, args.envs, args.read_batch_size) <= 0:
        parser.error("steps, stride, envs and read batch size must be positive")
    try:
        cfg = collection_config(Config.from_yaml(args.config), args.curriculum_stage, args.pool_size)
    except ValueError as error:
        parser.error(str(error))
    model = build_network(cfg, jax.random.PRNGKey(cfg.seed))
    model = eqx.tree_deserialise_leaves(args.checkpoint, model)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "config": cfg.to_dict(), "curriculum_stage": args.curriculum_stage,
              "export_steps": args.steps, "export_stride": args.stride,
              "upstream": json.loads((ROOT / "upstream-lock.json").read_text()),
              "note": "Warm-start descriptions are observable facts, not explanations of decisions."}
    splits = [("train", args.seed, args.envs),
              ("validation", args.seed + 1000, max(2, args.envs // 2))]
    if args.include_test or args.only_split == "test":
        splits.append(("test", args.seed + 2000, max(2, args.envs // 2)))
    if args.only_split:
        splits = [entry for entry in splits if entry[0] == args.only_split]
    for split, seed, count in splits:
        record[split] = collect_split(model, cfg, seed=seed, n_envs=count, steps=args.steps,
                                     stride=args.stride, output=args.output, split=split,
                                     read_batch_size=args.read_batch_size)
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
        raise AssertionError("Frozen player checkpoint changed during export")
    record["player_checkpoint_unchanged"] = True
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
