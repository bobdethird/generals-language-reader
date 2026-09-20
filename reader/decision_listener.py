"""Independent board-to-action observer with optional semantic English context.

Text features deliberately omit source coordinates, direction, troop split, and
the explicit action sentence. This tests descriptive context beyond transcription.
The observer never receives actor activations or action logits as inputs.
"""
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from reader.behavior import normalize_inputs, legal_actions
from reader.decisions import greedy_actions, decision_facts, decision_caption


def context_features(texts):
    patterns = (
        r"destination is friendly territory", r"destination is visible enemy territory",
        r"destination is neutral territory", r"destination ownership is unknown",
        r"moves closer to our general", r"moves farther from our general",
        r"keeps the same distance from our general", r"direction relative to our general is unknown",
    )
    result = np.asarray([[bool(re.search(pattern, text, re.I)) for pattern in patterns] for text in texts], np.float32)
    if not len(texts):
        return np.empty((0, 8), np.float32)
    # Contradictory context is unavailable, rather than an arbitrary new code.
    for start in (0, 4):
        invalid = result[:, start:start+4].sum(-1) > 1
        result[invalid, start:start+4] = 0
    return result


class DecisionListener(nn.Module):
    def __init__(self, width=384, layers=6):
        super().__init__()
        self.board = nn.Sequential(nn.Conv2d(38, 64, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(64, width, 3, stride=3), nn.GELU())
        self.history = nn.Sequential(nn.Linear(1024, width), nn.GELU(), nn.Linear(width, width))
        self.text = nn.Linear(8, width, bias=False)
        self.positions = nn.Parameter(torch.randn(1, 65, width) * .01)
        layer = nn.TransformerEncoderLayer(width, 6, 3*width, dropout=0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.moves = nn.Linear(width, 8*9)
        self.wait = nn.Linear(width, 1)

    def forward(self, board, temporal, text, legal):
        patches = self.board(board).flatten(2).transpose(1, 2)
        global_token = self.history(temporal.flatten(1)) + self.text(text)
        hidden = self.norm(self.context(torch.cat([global_token[:, None], patches], 1) + self.positions))
        move_logits = self.moves(hidden[:, 1:]).reshape(len(board), 8, 8, 8, 3, 3)
        move_logits = move_logits.permute(0, 3, 1, 4, 2, 5).reshape(len(board), -1)
        logits = torch.cat([move_logits, self.wait(hidden[:, 0])], -1)
        return logits.masked_fill(~legal, -1e9)


def load_listener_split(root, split, *, count=None, seed=0):
    root = Path(root)
    shards = json.loads((root / "manifest.json").read_text())[split]["shards"]
    rows = []
    for shard in shards:
        rows.extend(json.loads(line) for line in (root / shard["path"] / "captions.jsonl").read_text().splitlines())
    if count is None:
        selected = np.arange(len(rows))
    else:
        first, rest, seen = [], [], set()
        for index in np.random.default_rng(seed).permutation(len(rows)):
            group = rows[index]["map_id"]
            (rest if group in seen else first).append(index)
            seen.add(group)
        selected = np.sort((first + rest)[:count])
    n = len(selected)
    board = np.empty((n, 38, 24, 24), np.float32)
    history = np.empty((n, 2, 512), np.float32)
    legal = np.empty((n, 8*24*24+1), bool)
    targets = np.empty(n, np.int64)
    texts = []
    offset = destination = 0
    for shard in shards:
        indices = selected[(selected >= offset) & (selected < offset + shard["samples"])] - offset
        if len(indices):
            with np.load(root / shard["path"] / "samples.npz", allow_pickle=False) as arrays:
                observations = arrays["observations"][indices]
                temporal = arrays["temporal"][indices]
                teacher = arrays["logits"][indices]
                masks = arrays["masks"][indices]
            end = destination + len(indices)
            board[destination:end], history[destination:end] = normalize_inputs(observations, temporal)
            legal[destination:end] = legal_actions(masks).numpy()
            targets[destination:end] = greedy_actions(teacher)
            texts.extend(decision_caption(decision_facts(obs, action))
                         for obs, action in zip(observations, targets[destination:end]))
            destination = end
        offset += shard["samples"]
    if not legal[np.arange(n), targets].all():
        raise ValueError("Teacher selected action is not legal")
    return {"board": board, "history": history, "legal": legal, "target": targets,
            "context": context_features(texts), "captions": texts, "indices": selected.tolist(),
            "rows": [{**rows[int(i)], "game_id": rows[int(i)]["map_id"]} for i in selected]}
