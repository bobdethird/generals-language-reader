"""Independent behavior predictors and leakage-safe targets for a text reader.

Predictors receive only observations, visible history, legal masks, and text.
Player logits and sampled actions are targets, never predictor inputs. All
equivalent pass actions are combined; move-conditioned metrics avoid a pass-only
accuracy shortcut. The explanation reader itself receives only activations.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reader.captions import observation_facts

BEHAVIOR_PROMPT = (
    "Read the generals.io player's internal activation tokens. Describe the game "
    "situation and what behavior it suggests in short, ordinary English. Mention "
    "troop locations or pressure when supported. Do not give an exact move, action "
    "ID, or coordinate. Do not invent hidden enemies. Use at most four sentences."
)


def region(row, col, height, width):
    vertical = ("upper", "middle", "lower")[min(2, 3 * row // height)]
    horizontal = ("left", "center", "right")[min(2, 3 * col // width)]
    return f"{vertical} {horizontal}"


def grounding_caption(obs):
    """Automatic observation facts, without accessing player logits or actions.

    These are English grounding examples, not asserted reasons for a decision.
    Free-form generated text replaces these targets in behavior reward training.
    """
    h, w = obs.shape[-2:]
    own = obs[1]
    r, c = np.unravel_index(own.argmax(), own.shape)
    parts = [f"Largest friendly force is in the {region(r, c, h, w)}."]
    general = np.argwhere((obs[6] > 0) & (obs[10] > 0))
    if len(general):
        r, c = general[0]
        parts.append(f"Our general is in the {region(r, c, h, w)}.")
    enemy = obs[2]
    if enemy.max() > 0:
        r, c = np.unravel_index(enemy.argmax(), enemy.shape)
        parts.append(f"Largest visible enemy force is in the {region(r, c, h, w)}.")
    else:
        parts.append("No enemy troops are visible.")
    parts.append(observation_facts(obs).split(".")[0] + ".")
    return " ".join(parts)


def normalize_inputs(observations, temporal):
    obs = np.array(observations, dtype=np.float32, copy=True)
    channels = [0, 1, 2, 3, 14, 16, 17, 18, 19, 20] + list(range(24, obs.shape[1]))
    obs[:, channels] /= 50.0
    return obs, np.asarray(temporal, dtype=np.float32) / 50.0


def canonical_targets(logits):
    """Preserve the probability mass of every equivalent pass encoding."""
    cells = (logits.shape[-1] // 9)
    logits = torch.as_tensor(logits, dtype=torch.float32)
    merged = torch.cat([logits[:, :8 * cells],
                        torch.logsumexp(logits[:, 8 * cells:], -1, keepdim=True)], -1)
    return merged.softmax(-1)


def legal_actions(masks):
    directional = torch.as_tensor(masks, dtype=torch.bool).permute(0, 3, 1, 2).flatten(1)
    return torch.cat([directional, directional,
                      torch.ones((len(masks), 1), dtype=torch.bool)], -1)


def canonical_actions(actions, size):
    a = np.asarray(actions, dtype=np.int64)
    return np.where(a[:, 0] > 0, 8 * size * size,
                    (a[:, 3] + 4 * a[:, 4]) * size * size + a[:, 1] * size + a[:, 2])


class MovePredictor(nn.Module):
    def __init__(self, text_dim, temporal_dim=1024):
        super().__init__()
        self.board = nn.Sequential(nn.Conv2d(38, 64, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.history = nn.Sequential(nn.Linear(temporal_dim, 64), nn.GELU())
        self.text = nn.Sequential(nn.Linear(text_dim, 64), nn.Tanh())
        self.mix = nn.Sequential(nn.Conv2d(192, 64, 1), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.moves = nn.Conv2d(64, 8, 1)
        self.passing = nn.Linear(64, 1)

    def forward(self, observations, temporal, text, legal):
        board = self.board(observations)
        h = self.history(temporal.flatten(1))[:, :, None, None].expand_as(board)
        t = self.text(text)[:, :, None, None].expand_as(board)
        mixed = self.mix(torch.cat([board, h, t], 1))
        logits = torch.cat([self.moves(mixed).flatten(1),
                            self.passing(mixed.mean((2, 3)))], 1)
        return logits.masked_fill(~legal, -1e9)


def behavior_scores(logits, targets, sampled_actions=None):
    logq = F.log_softmax(logits, dim=-1)
    logp = targets.clamp_min(1e-30).log()
    full_kl = (targets * (logp - logq)).sum(-1)
    mass = targets[:, :-1].sum(-1)
    has_moves = mass > 1e-7
    move_p = targets[:, :-1] / mass[:, None].clamp_min(1e-7)
    move_logq = F.log_softmax(logits[:, :-1], dim=-1)
    move_kl = (move_p * (move_p.clamp_min(1e-30).log() - move_logq)).sum(-1)
    move_ce = -(move_p * move_logq).sum(-1)
    agreement = (logits[:, :-1].argmax(-1) == targets[:, :-1].argmax(-1)).float()
    result = {"full_kl": full_kl, "move_kl": move_kl, "move_ce": move_ce,
              "has_moves": has_moves, "move_top1": agreement,
              "pass_mae": (logq[:, -1].exp() - targets[:, -1]).abs()}
    if sampled_actions is not None:
        result["sampled_action_nll"] = -logq.gather(1, sampled_actions[:, None]).squeeze(1)
    return result


def distillation_loss(logits, targets):
    scores = behavior_scores(logits, targets)
    moves = scores["has_moves"].float()
    # Equal emphasis on total behavior and conditional movement; a large pass
    # mass must not hide an inability to reconstruct troop-moving decisions.
    return scores["full_kl"].mean() + (scores["move_kl"] * moves).sum() / moves.sum().clamp_min(1)


def eos_mask(tokens, eos_token_id):
    """Include the first EOS in the likelihood, ignore following padding."""
    is_eos = tokens == eos_token_id
    return (is_eos.long().cumsum(-1) - is_eos.long()) == 0


def group_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("Need equally sized groups of at least two candidates")
    grouped = rewards.reshape(-1, group_size)
    return ((grouped - grouped.mean(-1, keepdim=True)) * group_size / (group_size - 1)).flatten()


class BehaviorReader:
    """Frozen LM text generation and independent, text-only feature extraction."""
    def __init__(self, reader):
        self.reader = reader

    @torch.no_grad()
    def text_features(self, texts, batch_size=16):
        features = []
        r = self.reader
        for start in range(0, len(texts), batch_size):
            batch = [s if s.strip() else "No description available." for s in texts[start:start + batch_size]]
            ids = r.tokenizer(batch, padding=True, truncation=True, max_length=160,
                              return_tensors="pt").to(r.device)
            hidden = r.model.model(**ids, use_cache=False).last_hidden_state
            mask = ids.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            features.append((F.normalize(pooled, dim=-1) * pooled.shape[-1] ** 0.5).cpu())
        return torch.cat(features)

    @torch.no_grad()
    def sample(self, activations, *, max_new_tokens=80, sample=True):
        r = self.reader
        prefix = r.prefix(activations)
        tokens = r.model.generate(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=r.device),
            max_new_tokens=max_new_tokens, do_sample=sample,
            temperature=1.0, top_p=1.0, top_k=0,
            pad_token_id=r.tokenizer.pad_token_id, eos_token_id=r.tokenizer.eos_token_id)
        return tokens, r.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def token_log_probs(self, activations, tokens):
        r = self.reader
        prefix = r.prefix(activations)
        tokens = tokens.to(r.device)
        mask = eos_mask(tokens, r.tokenizer.eos_token_id)
        embeddings = r.model.get_input_embeddings()(tokens)
        attention = torch.cat([torch.ones(prefix.shape[:2], dtype=torch.bool, device=r.device), mask], 1)
        outputs = r.model(inputs_embeds=torch.cat([prefix, embeddings], 1),
                          attention_mask=attention, use_cache=False)
        logits = outputs.logits[:, prefix.shape[1] - 1:-1]
        selected = F.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return selected, mask
