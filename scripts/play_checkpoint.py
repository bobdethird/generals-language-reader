"""Play a frozen EMA checkpoint locally using Average Joe's human-game UI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(ROOT / ".cache/human-play"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

from reader.runtime import import_upstream, install_simulator_compat

import_upstream()
install_simulator_compat()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pygame

from config import Config
from evals.agent import Agent
from evals import eval_human
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation
from generals.core.action import compute_valid_move_mask
from networks import build_network, get_network_bundle, obs_to_array


def make_env(cfg):
    """Use the final curriculum difficulty and the training terrain settings."""
    stage = cfg.curriculum_stages[-1] if cfg.curriculum_stages else cfg

    def value(field):
        stage_value = getattr(stage, field, None)
        return getattr(cfg, field) if stage_value is None else stage_value

    return GeneralsEnv(
        grid_dims=(cfg.eval_grid_size, cfg.eval_grid_size), pad_to=cfg.pad_to,
        min_generals_distance=value("min_generals_distance"),
        max_generals_distance=value("max_generals_distance"),
        mountain_density_range=(cfg.mountain_density_min, cfg.mountain_density_max),
        castle_val_range=(value("castle_val_min"), value("castle_val_max")),
        num_cities_range=(value("num_cities_min"), value("num_cities_max")),
        truncation=cfg.truncation, pool_size=1,
    )


def smoke_test(agent, cfg, seed):
    env = make_env(cfg)
    pool, state = env.reset(jax.random.PRNGKey(seed))
    history = agent.init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    choose = eqx.filter_jit(agent.greedy_fn)
    times = []
    for _ in range(8):
        start = time.perf_counter()
        obs = get_observation(state, 1)
        augmented, history = agent.augment_fn(obs_to_array(obs), history)
        mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
        temporal = jnp.stack([history.opponent_army_history, history.opponent_land_history])
        action = choose(agent.network, augmented, mask, temporal)
        action.block_until_ready()
        times.append(time.perf_counter() - start)
        assert action.shape == (5,) and np.isfinite(np.asarray(action)).all()
        step, state = env.step(state, jnp.stack([eval_human.PASS_ACTION, action]), pool)
        state.time.block_until_ready()
        assert not bool(step.terminated)
    print(json.dumps({"smoke_test": "passed", "turns": 8,
                      "warm_decision_ms": 1000 * float(np.median(times[2:])),
                      "general_distance": [env.min_generals_distance, env.max_generals_distance],
                      "mountain_density": list(env.mountain_density_range)}), flush=True)


def play(agent, cfg, args):
    base_renderer = eval_human.Renderer
    base_make_env = eval_human._make_env

    class PlayRenderer(base_renderer):
        def __init__(self, properties):
            # SDL scales both pixels and mouse coordinates, retaining the
            # upstream board geometry while fitting the window on a laptop.
            original_set_mode = pygame.display.set_mode

            def scaled_mode(size, flags=0, *positional, **kwargs):
                return original_set_mode(size, flags | pygame.SCALED | pygame.RESIZABLE,
                                         *positional, **kwargs)

            pygame.display.set_mode = scaled_mode
            try:
                super().__init__(properties)
            finally:
                pygame.display.set_mode = original_set_mode
            from pygame._sdl2 import Window
            window = Window.from_display_module()
            width, height = self.screen.get_size()
            desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
            scale = min(1.0, (desktop_w - 100) / width, (desktop_h - 120) / height)
            window.size = (int(width * scale), int(height * scale))
            pygame.display.set_caption(f"Generals — {args.checkpoint.stem} — P to start")
            self.help_font = pygame.font.Font(None, 24)
            # The upstream loop consumes this before its first game tick.
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0))
            print("READY: game window open and paused. Press P to start.", flush=True)

        def render_stats(self):
            super().render_stats()
            lines = ["YOU: RED" if args.player == 0 else "YOU: BLUE", "Opponent:", args.checkpoint.stem,
                     "", "P: start / pause", "Click your tile to select", "W A S D: queue moves",
                     "Shift + W A S D: move half", "E: undo last move", "Q: clear queued moves",
                     "Space: deselect", "[ / ]: slower / faster", "V: reveal map (cheat)"]
            for i, line in enumerate(lines):
                text = self.help_font.render(line, True, (225, 230, 235))
                self.screen.blit(text, (self.display_grid_width + 12, 160 + 30 * i))

    eval_human._make_env = make_env
    eval_human.Renderer = PlayRenderer
    try:
        eval_human.run(agent, cfg, game_speed=args.speed, seed=args.seed,
                       human_player=args.player)
    finally:
        eval_human.Renderer = base_renderer
        eval_human._make_env = base_make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs/play-2000/L_7d_gae90_ema_2000.eqx")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/averagejoe-published.yaml")
    parser.add_argument("--speed", type=float, default=2, help="Game ticks per second")
    parser.add_argument("--seed", type=int, default=secrets.randbelow(2**31))
    parser.add_argument("--grid-size", type=int, default=23)
    parser.add_argument("--player", type=int, choices=[0, 1], default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Check real inference without opening a window")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    cfg = Config.from_yaml(args.config)
    if not cfg.min_grid_size <= args.grid_size <= cfg.max_grid_size:
        parser.error("grid size must be inside the trained size range")
    if not 0.5 <= args.speed <= 16:
        parser.error("speed must be between 0.5 and 16")
    cfg.eval_grid_size = args.grid_size
    print(f"Loading {args.checkpoint.name} locally on CPU...", flush=True)
    model = build_network(cfg, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.checkpoint, model)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))):
        raise ValueError("Checkpoint contains nonfinite weights")
    agent = Agent(model, cfg, get_network_bundle(cfg.network))
    output = ROOT / "runs/play-2000"
    output.mkdir(exist_ok=True, parents=True)
    record = {"checkpoint": str(args.checkpoint),
              "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "parameters": agent.param_count(), "seed": args.seed,
              "grid_size": args.grid_size, "human_player": args.player,
              "device": str(jax.devices()[0]), "frozen": True}
    (output / "last-game.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    os.chdir(output)  # Upstream game logs stay inside ignored run artifacts.
    if args.smoke_test:
        smoke_test(agent, cfg, args.seed)
    else:
        play(agent, cfg, args)


if __name__ == "__main__":
    main()
