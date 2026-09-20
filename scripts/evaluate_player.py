"""Evaluate a frozen EMA checkpoint on fresh paired maps against random play."""
import argparse
from dataclasses import replace
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

from reader.runtime import import_upstream, install_simulator_compat
import_upstream()
install_simulator_compat()
from config import Config
from generals.core.env import GeneralsEnv
from networks import build_network, get_network_bundle
from train.evaluations import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Network-only EMA checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=25044)
    parser.add_argument("--curriculum-stage", type=int, help="Evaluate this stage's environment, not the base config")
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2:
        parser.error("games must be positive and even (paired player positions)")
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = Config.from_yaml(args.config)
    if cfg.curriculum_stages:
        if args.curriculum_stage is None or not 0 <= args.curriculum_stage < len(cfg.curriculum_stages):
            parser.error("Curriculum runs require a valid --curriculum-stage")
        stage = cfg.curriculum_stages[args.curriculum_stage]
        stage_fields = ("min_generals_distance", "max_generals_distance", "castle_val_min",
                        "castle_val_max", "num_cities_min", "num_cities_max")
        cfg = replace(cfg, **{field: getattr(stage, field) for field in stage_fields if getattr(stage, field) is not None})
    model = build_network(cfg, jax.random.PRNGKey(cfg.seed))
    model = eqx.tree_deserialise_leaves(args.checkpoint, model)
    env = GeneralsEnv(min_grid_size=cfg.min_grid_size, max_grid_size=cfg.max_grid_size,
                      pad_to=cfg.pad_to, min_generals_distance=cfg.min_generals_distance,
                      max_generals_distance=cfg.max_generals_distance,
                      truncation=cfg.truncation, pool_size=1024,
                      castle_val_range=(cfg.castle_val_min, cfg.castle_val_max),
                      num_cities_range=(cfg.num_cities_min, cfg.num_cities_max),
                      mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max))
    pool_key, key = jax.random.split(jax.random.PRNGKey(args.seed))
    pool, _ = env.reset(pool_key)
    bundle = get_network_bundle(cfg.network)
    obs_state = bundle["init_obs_state"](cfg.pad_to, cfg.pad_to)
    n_maps = args.games // 2
    history = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_maps, *x.shape)), obs_state)
    wins, losses, draws, finished, both_wins, both_losses, split, _ = evaluate(
        env, model, key, cfg.truncation, n_maps, cfg.pad_to, history,
        bundle["augment_obs"], bundle["reset_obs_state"], bundle["greedy_action"], pool)
    result = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "seed": args.seed, "games": args.games, "wins": int(wins),
              "losses": int(losses), "draws": int(draws), "finished": int(finished),
              "won_both_seats_on_maps": int(both_wins), "lost_both_seats_on_maps": int(both_losses),
              "split_decisive_maps": int(split), "win_rate": int(wins) / args.games,
              "opponent": "upstream random legal-action agent", "action_selection": "greedy",
              "devices": [str(d) for d in jax.devices()], "truncation": cfg.truncation,
              "curriculum_stage": args.curriculum_stage,
              "generals_distance": [cfg.min_generals_distance, cfg.max_generals_distance],
              "scope": "Separate random-opponent evaluation; not a rating or broad strategic benchmark."}
    if int(finished) != args.games or int(wins + losses + draws) != args.games:
        raise AssertionError("Evaluation did not account for all paired games")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
