"""Detached activation readout matching the pinned Average Joe forward pass.

The player always uses its original forward pass. This copy is independently
checked against it before an activation dataset is exported.
"""
import jax
import jax.numpy as jnp

from reader.runtime import import_upstream
import_upstream()

from networks.common import normalize_observations, prepare_action_mask
from networks.transformer import _to_bf16


def read_activations(network, observation, temporal):
    net = _to_bf16(network) if network.use_bf16 else network
    obs = normalize_observations(observation)
    if net.use_bf16:
        obs, temporal = obs.astype(jnp.bfloat16), temporal.astype(jnp.bfloat16)
    patch = net.patch_size
    grid = net.pad_to // patch
    tokens = obs.reshape(net.n_channels, grid, patch, grid, patch)
    tokens = tokens.transpose(1, 3, 0, 2, 4).reshape(grid * grid, -1)
    tokens = jax.vmap(net.embedder)(tokens)
    history = net.temporal_encoder(temporal) + net.temporal_type_embed
    tokens = jnp.concatenate([net.value_token, history, tokens]) + net.pos_encoding
    for layer in net.transformer_layers:
        tokens = layer(tokens)
    tokens = jax.vmap(net.norm_out)(tokens)
    return jax.lax.stop_gradient(tokens.astype(jnp.float32))


def outputs_from_activations(network, tokens, mask):
    """Evaluate frozen original heads on a supplied activation reconstruction."""
    net = _to_bf16(network) if network.use_bf16 else network
    if net.use_bf16:
        tokens = tokens.astype(jnp.bfloat16)
    patch = net.patch_size
    grid = net.pad_to // patch
    logits = jax.vmap(net.policy_head)(tokens[3:]).astype(jnp.float32)
    logits = logits.reshape(grid, grid, 9, patch, patch)
    logits = logits.transpose(2, 0, 3, 1, 4).reshape(9, net.pad_to, net.pad_to)
    logits = logits + prepare_action_mask(mask, net.pad_to)
    value_raw = net.value_head(tokens[0]).astype(jnp.float32)
    value = (jnp.sum(jax.nn.softmax(value_raw) * network.bin_centers)
             if net.num_bins > 0 else value_raw[0])
    return logits.reshape(-1), value
