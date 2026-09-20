"""Export independent train/validation self-play sets from a frozen checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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


def collect_split(network, cfg, *, seed, n_envs, steps, stride, output, split):
    # Each split has independent map-generation and reset-pool seeds. Both seats
    # and every frame of a game stay together, including auto-reset episodes.
    env = GeneralsEnv(min_grid_size=cfg.min_grid_size, max_grid_size=cfg.max_grid_size,
                      pad_to=cfg.pad_to, min_generals_distance=cfg.min_generals_distance,
                      max_generals_distance=cfg.max_generals_distance,
                      truncation=cfg.truncation, pool_size=cfg.pool_size,
                      castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
                      num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
                      mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max))
    key, pool_key, state_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    pool, _ = env.reset(pool_key)
    states = jax.vmap(env.init_state)(jax.random.split(state_key, n_envs))
    bundle = get_network_bundle(cfg.network)
    initial = bundle["init_obs_state"](cfg.pad_to, cfg.pad_to)
    history = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_envs, *x.shape)), initial)
    _, rollout, _, _, _ = collect_rollout(
        states, env, network, key, steps, history, history, cfg.pad_to,
        win_lose_reward, bundle["augment_obs"], bundle["reset_obs_state"], cfg.gamma, pool)
    obs, masks, temporal, actions, _, _, _, _, terminated, truncated, _, _ = rollout
    done = np.asarray(terminated | truncated)[:, :n_envs]
    episode = np.cumsum(np.concatenate([np.zeros((1, n_envs), dtype=int), done[:-1]]), axis=0)
    frames = np.arange(0, steps, stride)
    flatten = lambda x: x[frames].reshape((-1, *x.shape[2:]))
    obs, masks, temporal, actions = map(flatten, (obs, masks, temporal, actions))

    @eqx.filter_jit
    def batch_read(net, o, m, t):
        h = jax.vmap(read_activations, in_axes=(None, 0, 0))(net, o, t)
        original_logits, original_values, _ = jax.vmap(net._forward)(o, m, t)
        read_logits, read_values = jax.vmap(outputs_from_activations, in_axes=(None, 0, 0))(net, h, m)
        return h, original_logits, original_values, read_logits, read_values

    activations, logits, values, read_logits, read_values = batch_read(network, obs, masks, temporal)
    np.testing.assert_allclose(read_logits, logits, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(read_values, values, atol=2e-5, rtol=2e-5)
    host_obs = np.asarray(obs, dtype=np.float32)
    prefix = output / split
    prefix.mkdir()
    np.savez_compressed(prefix / "samples.npz", activations=np.asarray(activations),
                        observations=host_obs, masks=np.asarray(masks),
                        temporal=np.asarray(temporal), actions=np.asarray(actions),
                        logits=np.asarray(logits), values=np.asarray(values))
    rows = []
    for index, o in enumerate(host_obs):
        frame, seat_index = divmod(index, 2 * n_envs)
        seat, lane = divmod(seat_index, n_envs)
        turn = int(frames[frame])
        rows.append({"index": index, "game_id": f"{split}:{seed}:{lane}:{episode[turn, lane]}",
                     "seat": seat, "turn": int(o[14, 0, 0]),
                     "caption": observation_facts(o), "caption_type": "observable_fact_warm_start"})
    (prefix / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"{split}: {len(rows)} snapshots; original/readout outputs agree", flush=True)
    return {"samples": len(rows), "map_seed": seed, "environments": n_envs,
            "head_readout_max_error": float(np.max(np.abs(np.asarray(read_logits) - np.asarray(logits))))}


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
    args = parser.parse_args()
    if min(args.steps, args.stride, args.envs) <= 0:
        parser.error("steps, stride, and envs must be positive")
    cfg = Config.from_yaml(args.config)
    model = build_network(cfg, jax.random.PRNGKey(cfg.seed))
    model = eqx.tree_deserialise_leaves(args.checkpoint, model)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "config": cfg.to_dict(), "upstream": json.loads((ROOT / "upstream-lock.json").read_text()),
              "note": "Warm-start descriptions are observable facts, not explanations of decisions."}
    splits = [("train", args.seed, args.envs),
              ("validation", args.seed + 1000, max(2, args.envs // 2))]
    if args.include_test:
        splits.append(("test", args.seed + 2000, max(2, args.envs // 2)))
    for split, seed, count in splits:
        record[split] = collect_split(model, cfg, seed=seed, n_envs=count, steps=args.steps,
                                     stride=args.stride, output=args.output, split=split)
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
