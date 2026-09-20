from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json

import jax
import numpy as np

from reader.runtime import import_upstream, install_simulator_compat
from reader.upstream_training import (make_cached_match, make_reference_evaluator,
                                       paired_match, refresh_curriculum_generation)
import_upstream()
install_simulator_compat()


def test_paired_matches_swap_outcomes_and_reuse_maps():
    calls = []
    key = jax.random.PRNGKey(42)
    def play(a, b, env, pool, games, truncation, match_key):
        calls.append((a, b, match_key))
        return (7, 2, 1) if a == "a" else (6, 3, 1)
    assert paired_match(play, "a", "b", None, None, 10, 512, key) == dict(wins=10, losses=8, draws=2)
    assert [(a, b) for a, b, _ in calls] == [("a", "b"), ("b", "a")]
    np.testing.assert_array_equal(calls[0][2], calls[1][2])


def test_reference_history_does_not_accumulate_candidate_results(tmp_path):
    calls = []
    def play(a, b, env, pool, games, truncation, key):
        calls.append((a.name, b.name))
        return (2, 1, 1)
    evaluate = make_reference_evaluator(play, tmp_path)
    refs = [SimpleNamespace(name="old"), SimpleNamespace(name="recent")]
    candidates = [SimpleNamespace(name="current")]
    matrix = {}
    first = evaluate(candidates, refs, matrix, None, None, 4, 512, jax.random.PRNGKey(1))
    second = evaluate(candidates, refs, matrix, None, None, 4, 512, jax.random.PRNGKey(2))
    assert len(calls) == 10  # One paired reference match, then four paired candidate matches.
    assert "current" not in matrix
    assert first == second
    assert first[1]["current"]["old"] == dict(wins=3, losses=3, draws=2)
    assert len((tmp_path / "evaluations.jsonl").read_text().splitlines()) == 2


def distances(states):
    states = jax.tree.map(np.asarray, states)
    result = []
    for ground, generals in zip(states.passable, states.general_positions):
        start, target = map(tuple, generals)
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            (r, c), distance = queue.popleft()
            if (r, c) == target:
                result.append(distance)
                break
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < ground.shape[0] and 0 <= cc < ground.shape[1] and ground[rr, cc] and (rr, cc) not in visited:
                    visited.add((rr, cc))
                    queue.append(((rr, cc), distance + 1))
        else:
            raise AssertionError("Disconnected generals")
    return result


def test_curriculum_refresh_changes_pooled_and_initial_games():
    from generals.core.env import GeneralsEnv
    env = GeneralsEnv(min_grid_size=8, max_grid_size=8, pad_to=9, pool_size=4,
                     min_generals_distance=2, max_generals_distance=3,
                     num_cities_range=(1, 2), castle_val_range=(13, 18))
    key = jax.random.PRNGKey(51)
    pool, _ = env.reset(key)
    def initializer(k):
        return jax.vmap(env.init_state)(jax.random.split(k, 4))
    keys = jax.random.split(key, jax.device_count())
    initial = jax.pmap(initializer)(keys)
    assert all(2 <= d <= 3 for d in distances(pool))
    assert all(2 <= d <= 3 for d in distances(jax.tree.map(lambda x: x[0], initial)))
    env.min_generals_distance, env.max_generals_distance = 6, 8
    new_initializer = refresh_curriculum_generation(env, initializer)
    harder_pool, _ = env.reset(key)
    harder_initial = new_initializer(keys)
    assert all(6 <= d <= 8 for d in distances(harder_pool))
    assert all(6 <= d <= 8 for d in distances(jax.tree.map(lambda x: x[0], harder_initial)))


def test_match_executor_reuses_compilation_with_different_weights():
    from config import Config
    from networks import build_network, get_network_bundle
    from generals.core.env import GeneralsEnv
    from evals.agent import Agent
    cfg = Config(pad_to=9, depth=1, embed_dim=16, n_head=2, ff_factor=2,
                 patch_size=3, network="history_transformer", value_loss="ce", num_bins=16)
    bundle = get_network_bundle(cfg.network)
    agents = [Agent(build_network(cfg, jax.random.PRNGKey(i)), cfg, bundle, name=str(i)) for i in range(2)]
    env = GeneralsEnv(min_grid_size=8, max_grid_size=8, pad_to=9, pool_size=4,
                     min_generals_distance=2, max_generals_distance=3, truncation=4,
                     num_cities_range=(1, 2), castle_val_range=(13, 18))
    pool, _ = env.reset(jax.random.PRNGKey(10))
    play = make_cached_match()
    result = paired_match(play, *agents, env, pool, 2, 4, jax.random.PRNGKey(20))
    assert sum(result.values()) == 4
    assert len(play.executors) == 1
