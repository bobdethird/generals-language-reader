import numpy as np
import pytest
import jax
import jax.numpy as jnp

from reader.runtime import import_upstream, install_jax_compat, install_simulator_compat
import_upstream()
from networks.transformer import HistoryTransformer
from reader.activations import read_activations, outputs_from_activations
from reader.captions import observation_facts


@pytest.mark.parametrize("bf16", [False, True])
def test_detached_readout_preserves_policy_and_value(bf16):
    model = HistoryTransformer(grid_size=9, patch_size=3, depth=2, embed_dim=32,
                               n_head=4, value_loss="ce", num_bins=32,
                               use_bf16=bf16, key=jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (38, 9, 9))
    history = jax.random.uniform(jax.random.PRNGKey(2), (2, 512)) * 100
    mask = jax.random.bernoulli(jax.random.PRNGKey(3), shape=(9, 9, 4))
    logits, value, _ = model._forward(obs, mask, history)
    hidden = read_activations(model, obs, history)
    actual_logits, actual_value = outputs_from_activations(model, hidden, mask)
    np.testing.assert_allclose(actual_logits, logits, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual_value, value, atol=2e-5, rtol=2e-5)
    # Language-readout updates cannot travel back into the player's inputs.
    gradient = jax.grad(lambda x: read_activations(model, x, history).sum())(obs)
    np.testing.assert_array_equal(gradient, 0)


def test_replicated_parameters_survive_pmap():
    install_jax_compat()
    tree = {"weights": jnp.arange(6).reshape(2, 3), "empty": None}
    replicated = jax.device_put_replicated(tree, jax.devices())
    result = jax.pmap(lambda t: t["weights"] + 1)(replicated)
    for replica in np.asarray(result):
        np.testing.assert_array_equal(replica, np.asarray(tree["weights"]) + 1)


def test_data_parallel_gradients_keep_replicas_synchronized():
    from functools import partial
    if jax.device_count() != 2:
        pytest.skip("Run with two devices to exercise collective gradient averaging")
    install_jax_compat()
    weights = jax.device_put_replicated(jnp.array(0.0), jax.devices())

    @partial(jax.pmap, axis_name="devices")
    def update(weight, target):
        grad = jax.grad(lambda w: 0.5 * (w - target) ** 2)(weight)
        return weight - 0.1 * jax.lax.pmean(grad, "devices")

    updated = update(weights, jnp.array([1.0, 3.0]))
    np.testing.assert_allclose(updated, [0.2, 0.2], rtol=1e-6)
    np.testing.assert_allclose(update(updated, jnp.array([1.0, 3.0])),
                               [0.38, 0.38], rtol=1e-6)


def test_simulator_alias_updates_actual_configuration():
    install_simulator_compat()
    from generals.core.env import GeneralsEnv
    env = GeneralsEnv(num_cities_range=(1, 2))
    assert env.num_cities_range == (1, 2)
    env.num_cities_range = (3, 4)
    assert env.num_castles_range == (3, 4)


def test_caption_does_not_claim_unseen_enemy_general():
    obs = np.zeros((38, 9, 9))
    obs[6, 3, 4] = 1  # own general
    obs[10, 3, 4] = 1
    obs[17] = 20
    obs[19] = 40
    caption = observation_facts(obs)
    assert "Enemy army is larger." in caption
    assert "No enemy troops are visible." in caption
    assert "Enemy general has not been located." in caption
    obs[6, 7, 8] = 1  # previously discovered enemy general, now possibly fogged
    assert "Enemy general has been located." in observation_facts(obs)
