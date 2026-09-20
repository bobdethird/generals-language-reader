from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reader.published_training import global_batch, make_router, patch_fixed_batch


def test_recipe_is_byte_identical_to_pinned_upstream():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'configs/averagejoe-published.yaml').read_bytes() == (
        root / 'vendor/averagejoe/configs/custom/L_7d_gae90.yaml').read_bytes()


@pytest.mark.parametrize('local_samples', [16, 8192])
def test_global_selection_routes_each_top_example_once(local_samples):
    d = jax.device_count()
    if d < 2:
        pytest.skip('Run with XLA_FLAGS=--xla_force_host_platform_device_count=2')
    # All high advantages are on the last device: per-device top-k would fail.
    ids = jnp.arange(d * local_samples).reshape(d, 4, local_samples//4)
    advs = ids.astype(jnp.float32) - 1
    batch = (ids[..., None], (ids % 2 == 0)[..., None], advs)
    routed, indices, _, _ = global_batch(batch, advs, jax.random.PRNGKey(3), 0.25, 2)
    actual_ids = np.asarray(routed[0]).reshape(-1)
    wanted = np.argsort(np.abs(np.asarray(advs).reshape(-1)))[-d * local_samples//4:]
    np.testing.assert_array_equal(np.sort(actual_ids), np.sort(wanted))
    np.testing.assert_array_equal(np.asarray(routed[1]).reshape(-1), actual_ids % 2 == 0)
    np.testing.assert_array_equal(np.asarray(routed[2]).reshape(-1), actual_ids - 1)
    assert indices.shape == (d, local_samples//4)
    assert make_router(16) is make_router(16)


def test_fixed_batch_patch_fails_closed_on_upstream_changes():
    with pytest.raises(RuntimeError):
        patch_fixed_batch('def train(): pass')


@pytest.mark.parametrize('batch_mode', ['fixed_global', 'native', 'native4'])
def test_distributed_masked_update(batch_mode):
    if jax.device_count() < 2:
        pytest.skip('Needs two CPU devices')
    if batch_mode == 'native4' and jax.device_count() != 8:
        pytest.skip('Needs eight CPU devices for four-to-eight equivalence')
    import equinox as eqx
    import optax
    from reader.runtime import import_upstream, install_simulator_compat
    import_upstream()
    install_simulator_compat()
    from reader.published_training import install_published_support
    import train.ppo as ppo
    original = Path(ppo.__file__).read_text()
    original_update = original[original.index('def ppo_update('):original.index('# ---- Training Loop ----')]
    namespace = dict(vars(ppo))
    exec(compile(original_update, '<Test pinned original PPO>', 'exec'), namespace)
    reference_update = namespace['ppo_update']
    install_published_support(batch_mode)
    assert ('_global_batch' in ppo.train.__code__.co_names) == (batch_mode == 'fixed_global')

    class TinyPolicy(eqx.Module):
        policy_head: jax.Array
        value_head: jax.Array
        def __call__(self, o, mask, temporal, key, action):
            logits = self.policy_head * o[0]
            probabilities = jax.nn.softmax(logits)
            logs = jax.nn.log_softmax(logits)
            value = jnp.dot(self.value_head, o)
            return action, value, logs[action[0]], -jnp.sum(probabilities * logs), value, probabilities

    network = TinyPolicy(jnp.array([.1, -.2]), jnp.array([.2, .3]))
    optimizer = optax.chain(optax.clip_by_global_norm(.267), optax.adam(.001))
    opt_state = optimizer.init(network)
    obs = jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 10
    actions = (jnp.arange(8) % 2)[:, None]
    old_lps = jax.vmap(lambda o, a: network(o, None, None, None, a)[2])(obs, actions)
    batch = (obs, jnp.ones((8, 1), dtype=jnp.bool_), jnp.ones((8, 1)), actions,
             old_lps, jnp.linspace(-1, 1, 8), jnp.linspace(.8, -.8, 8),
             jnp.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.float32))

    def update(d, update_fn=None):
        update_fn = update_fn or ppo.ppo_update
        shaped = jax.tree.map(lambda x: x.reshape(d, 1, 8//d, *x.shape[1:]), batch)
        def step(net, state, data, key, idx):
            updated, state, metrics = update_fn(net, state, data, optimizer, key, .2, .5, .01, 8//d,
                lambda value, ret: .5*(value-ret)**2, idx,
                magnet_fn=lambda o, m: jnp.ones(2)/2)
            if batch_mode == 'native4':
                metrics = jax.lax.pmean(metrics, 'devices')
            return updated, state, metrics
        fn = jax.pmap(step, axis_name='devices', devices=jax.devices()[:d])
        replicated = lambda tree: jax.tree.map(lambda x: jnp.stack([x]*d), tree)
        return fn(replicated(network), replicated(opt_state), shaped,
                  jax.random.split(jax.random.PRNGKey(1), d),
                  jnp.tile(jnp.arange(8//d), (d, 1)))
    if batch_mode == 'native4':
        expected, actual = update(4, reference_update), update(8)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            np.testing.assert_allclose(np.repeat(np.asarray(a), 2, axis=0), np.asarray(b),
                                       rtol=2e-5, atol=2e-6)
    elif batch_mode == 'native':
        # In particular, retain upstream's mean of local masked means even with
        # unequal mask counts. Check parameters, Adam and reported metrics.
        expected, actual = update(2, reference_update), update(2)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    else:
        first, second = update(1), update(2)
        for expected, actual in zip(jax.tree.leaves(first[:2]), jax.tree.leaves(second[:2])):
            np.testing.assert_allclose(np.asarray(actual), np.repeat(np.asarray(expected), 2, axis=0),
                                       rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('count', [4, 8])
def test_native_batch_accounting(count):
    from reader.published_training import batch_settings
    native = batch_settings(count)
    assert native['per_device_envs'] == 512 and native['per_device_minibatch'] == 1024
    assert native['global_envs'] == 512*count
    assert native['global_minibatch'] == 1024*count
    assert native['player_samples_per_iteration'] == 524288*count
    assert native['adam_updates_per_iteration'] == 128
    fixed = batch_settings(count, 'fixed_global')
    assert fixed['global_envs'] == 512 and fixed['global_minibatch'] == 1024
    reference = batch_settings(count, 'native4')
    assert reference['global_envs'] == 2048 and reference['global_minibatch'] == 4096
    assert reference['player_samples_per_iteration'] == 2097152
    assert reference['per_device_envs'] == 2048//count


@pytest.mark.parametrize('local_samples', [16, 8192])
def test_eight_gpu_selection_preserves_four_upstream_top_k_sets(local_samples):
    if jax.device_count() != 8:
        pytest.skip('Needs eight CPU devices')
    from reader.published_training import reference_four_batch
    ids = jnp.arange(8*local_samples).reshape(8, 4, local_samples//4)
    advs = ids.astype(jnp.float32) - 1
    batch = (ids[..., None], (ids % 2 == 0)[..., None], advs)
    routed, indices, _, _ = reference_four_batch(batch, advs, jax.random.PRNGKey(7), .25, 2)
    actual = np.asarray(routed[0]).reshape(4, -1)
    reference_advs = np.asarray(advs).reshape(4, -1)
    for group in range(4):
        chosen = np.argsort(np.abs(reference_advs[group]))[-local_samples//2:]
        expected = chosen + group*2*local_samples
        np.testing.assert_array_equal(np.sort(actual[group]), np.sort(expected))
    np.testing.assert_array_equal(np.asarray(routed[1]).reshape(-1), actual.reshape(-1) % 2 == 0)
    np.testing.assert_array_equal(np.asarray(routed[2]).reshape(-1), actual.reshape(-1)-1)
    assert indices.shape == (8, local_samples//4)


@pytest.mark.parametrize('reason', ['deadline', 'configured_iterations'])
def test_stop_saves_weights_optimizer_and_ema_before_exit(tmp_path, monkeypatch, reason):
    import equinox as eqx
    import json
    import optax
    import time
    from types import SimpleNamespace
    from reader.published_training import iteration_done
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('AVERAGEJOE_DEADLINE', str(time.time() + (60 if reason == 'deadline' else 3600)))
    offset = 264
    local_iteration = 7 if reason == 'deadline' else 30000-offset
    network = {'weights': jnp.array([1., 2.])}
    optimizer = optax.adam(.001).init(network)
    ema = {'weights': jnp.array([.8, 1.9])}
    state = dict(it=local_iteration-1, m={k: jnp.array(1.) for k in
        ('total_loss', 'policy_loss', 'value_loss', 'grad_norm', 'approx_kl')},
        current_stage_idx=2, elapsed=.5, t_rollout=.2, t_ppo=.3,
        samples_per_iter=2097152, cfg=SimpleNamespace(num_iters=30000-offset,
            iteration_offset=offset, save_every=500),
        ckpt_dir=str(tmp_path), _get_network=lambda: network, _get_opt_state=lambda: optimizer,
        ema_params=ema, static={'weights': None}, last_eval_wr=.7)
    with pytest.raises(SystemExit) as stop:
        iteration_done(state)
    assert stop.value.code == 0
    progress = json.loads((tmp_path/'progress.json').read_text())
    assert progress['stop_reason'] == reason and progress['iteration'] == local_iteration+offset
    restored = eqx.tree_deserialise_leaves(progress['checkpoint'], (network, optimizer))
    restored_ema = eqx.tree_deserialise_leaves(progress['ema_checkpoint'], ema)
    for wanted, actual in zip(jax.tree.leaves((network, optimizer, ema)), jax.tree.leaves((*restored, restored_ema))):
        np.testing.assert_array_equal(wanted, actual)


@pytest.mark.parametrize('target', [30000, 100000])
def test_resume_preserves_recipe_and_continues_original_stage(target):
    from ruamel.yaml import YAML
    from reader.published_training import resume_settings
    root=Path(__file__).resolve().parents[1]
    original=YAML(typ='safe').load((root/'configs/averagejoe-published.yaml').read_text())
    progress=dict(iteration=137,stage=2,checkpoint='checkpoints/raw.eqx',ema_checkpoint='checkpoints/ema.eqx')
    result=resume_settings(original,progress,target,'/runs/prior')
    assert result['num_iters']+result['iteration_offset']==target
    assert result['curriculum']==original['curriculum'][2:]
    assert result['curriculum'][0]['max_generals_distance']==13
    state_fields={'num_iters','iteration_offset','curriculum','init_checkpoint','ema_checkpoint'}
    assert {k:v for k,v in result.items() if k not in state_fields} == {
        k:v for k,v in original.items() if k not in state_fields}
    assert original['init_checkpoint']=='' and len(original['curriculum'])==5
