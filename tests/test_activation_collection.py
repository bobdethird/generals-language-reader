import numpy as np
import pytest

from scripts.collect_activations import collection_config, episode_ids, pool_start_indices
from config import Config


def test_export_requires_and_applies_actual_curriculum_stage():
    cfg = Config.from_yaml("configs/averagejoe-published.yaml")
    with pytest.raises(ValueError, match="explicit"):
        collection_config(cfg)
    actual = collection_config(cfg, 4, 196)
    assert (actual.min_generals_distance, actual.max_generals_distance) == (17, 28)
    assert actual.pool_size == 196
    assert actual.embed_dim == 448 and actual.depth == 7
    assert cfg.min_generals_distance == 4 and cfg.pool_size == 200000
    with pytest.raises(ValueError, match="every size"):
        collection_config(cfg, 4, 48)


def test_episode_groups_change_after_completion_and_are_shared_by_seats():
    terminated = np.zeros((5, 4), dtype=bool)
    truncated = np.zeros_like(terminated)
    terminated[1, [0, 2]] = True
    truncated[3, [1, 3]] = True
    actual = episode_ids(terminated, truncated, 2)
    np.testing.assert_array_equal(actual, [[0, 0], [0, 0], [1, 0], [1, 0], [1, 1]])
    with pytest.raises(ValueError, match="boolean"):
        episode_ids(truncated, np.full((5, 4), -1), 2)


def test_pool_sections_do_not_share_resets():
    starts, width = pool_start_indices(6272, 64)
    assert width == 98
    assert len(set((starts[:, None] + np.arange(width)).ravel())) == 6272
    with pytest.raises(ValueError, match="two pool entries"):
        pool_start_indices(64, 64)


def test_actual_upstream_rollout_boundary_layout():
    import jax
    import jax.numpy as jnp
    from generals.core.env import GeneralsEnv
    from networks import build_network, get_network_bundle
    from train.rollout_selfplay import collect_rollout
    from train.rewards import win_lose_reward

    cfg = Config.from_yaml("configs/local-smoke.yaml")
    env = GeneralsEnv(grid_dims=(9, 9), pad_to=9, min_generals_distance=2,
                      max_generals_distance=5, pool_size=4, truncation=3,
                      num_cities_range=(0, 1))
    pool, _ = env.reset(jax.random.PRNGKey(1))
    states = jax.vmap(env.init_state)(jax.random.split(jax.random.PRNGKey(2), 2))
    net = build_network(cfg, jax.random.PRNGKey(3))
    bundle = get_network_bundle(cfg.network)
    history = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, *x.shape)),
                           bundle["init_obs_state"](9, 9))
    _, rollout, _, _, _ = collect_rollout(
        states, env, net, jax.random.PRNGKey(4), 4, history, history, 9,
        win_lose_reward, bundle["augment_obs"], bundle["reset_obs_state"], cfg.gamma, pool)
    np.testing.assert_array_equal(episode_ids(rollout[8], rollout[9], 2),
                                  [[0, 0], [0, 0], [0, 0], [1, 1]])
