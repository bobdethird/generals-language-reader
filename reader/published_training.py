"""Deadline/resume instrumentation for native upstream training.

The optional fixed_global mode preserves the earlier scaling experiments.
Native mode leaves per-device batches, top-k and PPO loss reductions upstream.
"""
import json
import math
import os
from pathlib import Path
import time
from functools import lru_cache, partial


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"Pinned upstream changed: {before[:100]}")
    return source.replace(before, after)


def batch_settings(count, mode="native"):
    if count not in (1, 2, 4, 8) or mode not in ("native", "fixed_global", "native4"):
        raise ValueError("Unsupported GPU count or batch mode")
    if mode == 'native4' and count not in (4, 8):
        raise ValueError('native4 uses four logical replicas on four or eight GPUs')
    divisor = count if mode == "fixed_global" else count // 4 if mode == 'native4' else 1
    local_envs, local_minibatch = 512 // divisor, 1024 // divisor
    return dict(batch_mode=mode, per_device_envs=local_envs,
                per_device_minibatch=local_minibatch, global_envs=count*local_envs,
                global_minibatch=count*local_minibatch, rollout_steps=512,
                player_samples_per_iteration=count*local_envs*512*2,
                adam_updates_per_iteration=128)


def resume_settings(recipe, progress, target, directory):
    """Resume learner state at its original curriculum stage and schedule step."""
    from copy import deepcopy
    iteration, stage = int(progress['iteration']), int(progress['stage'])
    if not 0 < iteration < target or not 0 <= stage < len(recipe['curriculum']):
        raise ValueError('Invalid resume iteration or curriculum stage')
    result = deepcopy(recipe)
    result.update(num_iters=target-iteration, iteration_offset=iteration,
                  init_checkpoint=str(Path(directory)/progress['checkpoint']),
                  ema_checkpoint=str(Path(directory)/progress['ema_checkpoint']),
                  curriculum=result['curriculum'][stage:])
    return result


@lru_cache(maxsize=16)
def make_router(total_local, group_size=None):
    import jax
    import jax.numpy as jnp

    groups = ([list(range(i, i+group_size)) for i in range(0, jax.device_count(), group_size)]
              if group_size else None)
    @partial(jax.pmap, axis_name="devices")
    def route(local_batch, requests):
        all_requests = jax.lax.all_gather(requests, "devices", axis_index_groups=groups)
        chunk_size = math.gcd(requests.shape[0], 1024)
        chunks = all_requests.reshape(all_requests.shape[0], -1, chunk_size).transpose(1, 0, 2)
        owner_id = jax.lax.axis_index("devices")
        def gather(array):
            flat = array.reshape(total_local, *array.shape[2:])
            def gather_chunk(indices):
                owner = indices // total_local == owner_id
                selected = flat[indices % total_local]
                mask = owner.reshape(*owner.shape, *([1] * (selected.ndim - 2)))
                selected = jnp.where(mask, selected, jnp.zeros((), selected.dtype))
                collective = selected.astype(jnp.int32) if selected.dtype == jnp.bool_ else selected
                result = jax.lax.psum_scatter(collective, "devices", scatter_dimension=0,
                                            axis_index_groups=groups)
                return result.astype(selected.dtype)
            # Keep each collective below giant multi-GiB tensor/indexing limits.
            routed = jax.lax.map(gather_chunk, chunks)
            return routed.reshape(1, requests.shape[0], *array.shape[2:])
        return jax.tree.map(gather, local_batch)
    return route


def shard_leading_axis(tree):
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    sharding = NamedSharding(Mesh(np.array(jax.devices()), ("devices",)), PartitionSpec("devices"))
    return jax.tree.map(lambda value: jax.device_put(value, sharding), tree)


def global_batch(batch, advs, key, fraction, local_minibatch):
    """Route the globally selected examples to devices using reduce-scatter.

    Only the selected data are exchanged; the entire rollout is never gathered.
    The initial global permutation makes subsequent independent local shuffles
    equivalent in distribution to sampling global minibatches without replacement.
    """
    import jax
    import jax.numpy as jnp
    devices = advs.shape[0]
    total_local = advs.shape[1] * advs.shape[2]
    keep = int(advs.size * fraction)
    keep = keep // (devices * local_minibatch) * (devices * local_minibatch)
    # The CUDA SPMD lowering of a large sharded top_k crashes in JAX 0.11.2.
    # Gather only the ~2 MiB of scalar advantages, then sort on one device.
    # Observations stay distributed and are routed below after selection.
    flat_advs = jax.device_put(advs, jax.devices()[0]).reshape(-1)
    _, indices = jax.lax.top_k(jnp.abs(flat_advs), keep)
    chosen = flat_advs[indices]
    key = jax.device_put(key, jax.devices()[0])
    indices = jax.random.permutation(key, indices).reshape(devices, keep // devices)

    routed = make_router(total_local)(shard_leading_axis(batch), shard_leading_axis(indices))
    sample_indices = jnp.broadcast_to(jnp.arange(keep // devices), (devices, keep // devices))
    return routed, shard_leading_axis(sample_indices), float(chosen.std()), float(jnp.abs(chosen).mean())


def reference_four_batch(batch, advs, key, fraction, local_minibatch):
    """Split each of four upstream replicas across two GPUs without changing top-k.

    Selection stays local to the original four logical replicas. A uniform
    permutation partitions each selected set over its GPU pair; subsequent local
    permutations retain the upstream distribution of global minibatches.
    """
    import jax
    import jax.numpy as jnp
    physical = advs.shape[0]
    if physical != 8:
        raise ValueError('The split four-replica path requires eight devices')
    total_local = advs.shape[1] * advs.shape[2]
    per_gpu_keep = int(total_local*fraction)//local_minibatch*local_minibatch
    logical_advs = jax.device_put(advs, jax.devices()[0]).reshape(4, -1)
    _, indices = jax.lax.top_k(jnp.abs(logical_advs), 2*per_gpu_keep)
    chosen = jnp.take_along_axis(logical_advs, indices, axis=1)
    key = jax.device_put(key, jax.devices()[0])
    indices = jax.vmap(jax.random.permutation)(jax.random.split(key, 4), indices)
    indices = indices + jnp.arange(4)[:, None] * (2*total_local)
    indices = indices.reshape(8, per_gpu_keep)
    routed = make_router(total_local, group_size=2)(shard_leading_axis(batch), shard_leading_axis(indices))
    sample_indices = jnp.broadcast_to(jnp.arange(per_gpu_keep), (8, per_gpu_keep))
    return routed, shard_leading_axis(sample_indices), float(chosen.std()), float(jnp.abs(chosen).mean())


def patch_fixed_batch(source):
    before = """        sample_idx = _compute_top_idx(advs)
        advs_flat = advs.reshape(num_devices, -1)
        filtered_advs = jnp.take_along_axis(advs_flat, sample_idx, axis=1)
        filtered_adv_std = float(filtered_advs.std())
        filtered_adv_mean = float(jnp.abs(filtered_advs).mean())

        batch = (obs, masks, temporal, actions, lps, advs, rets, train_mask)"""
    after = """        batch = (obs, masks, temporal, actions, lps, advs, rets, train_mask)
        if num_devices > 1:
            batch, sample_idx, filtered_adv_std, filtered_adv_mean = _global_batch(
                batch, advs, jrandom.fold_in(keys[0], 9173), cfg.adv_top_frac, cfg.minibatch_size)
        else:
            sample_idx = _compute_top_idx(advs)
            advs_flat = advs.reshape(num_devices, -1)
            filtered_advs = jnp.take_along_axis(advs_flat, sample_idx, axis=1)
            filtered_adv_std = float(filtered_advs.std())
            filtered_adv_mean = float(jnp.abs(filtered_advs).mean())"""
    return replace_once(source, before, after)


def iteration_done(state):
    import equinox as eqx
    import jax
    local_iteration = state["it"] + 1
    iteration = local_iteration + getattr(state["cfg"], "iteration_offset", 0)
    stage = state["current_stage_idx"] + int(os.environ.get("AVERAGEJOE_STAGE_OFFSET", "0"))
    metrics = state["m"]
    if not all(math.isfinite(float(metrics[name])) for name in
               ("total_loss", "policy_loss", "value_loss", "grad_norm", "approx_kl")):
        raise FloatingPointError(f"Non-finite training update at iteration {iteration}")
    now = time.time()
    row = dict(iteration=iteration, timestamp=now, stage=stage,
               update_seconds=state["elapsed"], rollout_seconds=state["t_rollout"],
               ppo_seconds=state["t_ppo"], samples=state["samples_per_iter"],
               gpu_memory=[d.memory_stats() for d in jax.devices()])
    with Path("iteration-timing.jsonl").open("a") as file:
        file.write(json.dumps(row) + "\n")
    deadline = float(os.environ["AVERAGEJOE_DEADLINE"])
    # Reserve two minutes to serialize/commit at a safe completed-update boundary.
    reason = "handoff" if Path("checkpoint-and-stop").exists() else "deadline" if now >= deadline - 120 else (
        "configured_iterations" if local_iteration >= state["cfg"].num_iters else None)
    if iteration % state["cfg"].save_every == 0 or reason:
        path = Path(state["ckpt_dir"])
        raw = path / "published_latest.eqx"
        ema = path / "published_latest_ema.eqx"
        eqx.tree_serialise_leaves(str(raw) + ".tmp", (state["_get_network"](), state["_get_opt_state"]()))
        eqx.tree_serialise_leaves(str(ema) + ".tmp", eqx.combine(state["ema_params"], state["static"]))
        Path(str(raw) + ".tmp").replace(raw)
        Path(str(ema) + ".tmp").replace(ema)
        progress = dict(iteration=iteration, stage=stage,
                        last_eval_win_rate=state["last_eval_wr"], stop_reason=reason,
                        checkpoint=str(raw), ema_checkpoint=str(ema), timestamp=now)
        Path("progress.json.tmp").write_text(json.dumps(progress, indent=2) + "\n")
        Path("progress.json.tmp").replace("progress.json")
        print(f"PUBLISHED_CHECKPOINT {iteration}: {reason or 'periodic'}", flush=True)
    if reason:
        print(f"PUBLISHED_STOP {reason} after {iteration} complete iterations", flush=True)
        raise SystemExit(0)


def install_published_support(batch_mode=None):
    import jax
    batch_mode = batch_mode or os.environ.get("AVERAGEJOE_BATCH_MODE", "native")
    batch_settings(jax.device_count(), batch_mode)
    split_reference = batch_mode == 'native4' and jax.device_count() == 8
    import train.ppo as ppo
    from reader.upstream_training import install_training_support
    install_training_support(paired_references=False)
    # inspect the original pinned file: the curriculum wrapper has a dynamic filename.
    original = Path(ppo.__file__).read_text()
    source = original[original.index("def train("):]
    from reader.upstream_training import patched_train_source
    source = patched_train_source(source)
    if batch_mode == "fixed_global":
        source = patch_fixed_batch(source)
    elif split_reference:
        source = patch_fixed_batch(source).replace('_global_batch(', '_reference_four_batch(')
    source = replace_once(source, '            it, cfg, eval_freq, network, ema_params, static,',
                          '            it + iter_offset, cfg, eval_freq, network, ema_params, static,')
    source = replace_once(source,
        'if cfg.reset_pool_every > 0 and it > 0 and it % cfg.reset_pool_every == 0:',
        'if cfg.reset_pool_every > 0 and it + iter_offset > 0 and (it + iter_offset) % cfg.reset_pool_every == 0:')
    source = replace_once(source, '        logger.log(it + 1, log_metrics)',
                          '        logger.log(it + 1 + iter_offset, log_metrics)')
    source = source.replace('Iter {it + 1:3d}/{cfg.num_iters}',
                            'Iter {it + 1 + iter_offset:3d}/{cfg.num_iters + iter_offset}')
    for field in ('ckpt_every', 'save_every'):
        source = replace_once(source, f'if (it + 1) % cfg.{field} == 0:',
                              f'if (it + 1 + iter_offset) % cfg.{field} == 0:')
    source = source.replace('{it + 1}.eqx', '{it + 1 + iter_offset}.eqx')
    for marker, label in (
        ('        # Collect rollout — pmapped across devices', 'rollout'),
        ('        # GAE advantages (per-device)', 'advantages'),
        ('        # Compute sample indices', 'sample selection'),
        ('        for _ in range(cfg.num_epochs):', 'optimizer update'),
    ):
        source = replace_once(source, marker,
            f'        if it == 0: print("PUBLISHED_PHASE {label}", flush=True)\n' + marker)
    source = replace_once(source,
        "        # Free large arrays to prevent BFC allocator fragmentation on next rollout",
        # locals() on Python 3.12 leaves a frame-owned dict retaining the whole
        # 21-GiB observation rollout after upstream deletes its large arrays.
        "        _published_iteration_done(dict(it=it, m=m, current_stage_idx=current_stage_idx,\n"
        "            elapsed=elapsed, t_rollout=t_rollout, t_ppo=t_ppo,\n"
        "            samples_per_iter=samples_per_iter, cfg=cfg, ckpt_dir=ckpt_dir,\n"
        "            _get_network=_get_network, _get_opt_state=_get_opt_state,\n"
        "            ema_params=ema_params, static=static, last_eval_wr=last_eval_wr))\n\n"
        "        # Free large arrays to prevent BFC allocator fragmentation on next rollout")
    ppo._global_batch = global_batch
    ppo._reference_four_batch = reference_four_batch
    ppo._published_iteration_done = iteration_done
    exec(compile(source, "<Published AverageJoe instrumentation>", "exec"), vars(ppo))
    # Always start from the pinned function, including when tests switch modes.
    update_source = original[original.index("def ppo_update("):original.index("# ---- Training Loop ----")]
    if batch_mode == "fixed_global":
        # Preserve the old experimental global masked mean only in that mode.
        update_source = replace_once(update_source,
            "            mean_loss = masked_losses.sum() / jnp.maximum(mb_mask.sum(), 1.0)",
            "            count = jax.lax.psum(mb_mask.sum(), axis_name='devices')\n"
            "            devices = jax.lax.psum(1.0, axis_name='devices')\n"
            "            mean_loss = devices * masked_losses.sum() / jnp.maximum(count, 1.0)")
    elif split_reference:
        groups = [[i, i+1] for i in range(0, 8, 2)]
        update_source = replace_once(update_source,
            "            mean_loss = masked_losses.sum() / jnp.maximum(mb_mask.sum(), 1.0)",
            f"            count = jax.lax.psum(mb_mask.sum(), 'devices', axis_index_groups={groups!r})\n"
            "            mean_loss = 2 * masked_losses.sum() / jnp.maximum(count, 1.0)")
        update_source = replace_once(update_source, '            return mean_loss, stats',
            "            for name in stats:\n"
            "                if name.startswith('max_'):\n"
            f"                    stats[name] = jax.lax.pmax(jax.lax.stop_gradient(stats[name]), 'devices', axis_index_groups={groups!r})\n"
            "                elif name.startswith('min_'):\n"
            f"                    stats[name] = jax.lax.pmin(jax.lax.stop_gradient(stats[name]), 'devices', axis_index_groups={groups!r})\n"
            "            return mean_loss, stats")
    exec(compile(update_source, "<Published global masked loss>", "exec"), vars(ppo))
    print(f"Published recipe: batch_mode={batch_mode}; deadline saves enabled.", flush=True)
