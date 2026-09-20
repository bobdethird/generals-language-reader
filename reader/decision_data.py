"""Add decision targets to detached snapshot inputs without leaking labels."""
import json
from pathlib import Path
import numpy as np
from reader.spatial_data import load_sharded_split
from reader.decisions import greedy_actions, decision_facts, decision_caption


def pack_legal_masks(hidden, masks):
    hidden, masks = np.asarray(hidden), np.asarray(masks)
    if hidden.shape[1:] != (67, 448) or masks.shape != (len(hidden), 24, 24, 4):
        raise ValueError("Unexpected activation or legal-mask shape")
    packed = np.zeros((len(hidden), 67, 484), np.float32)
    packed[..., :448] = hidden
    packed[:, 3:, 448:] = masks.reshape(-1, 8, 3, 8, 3, 4).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 64, 36)
    return packed


def load_decision_split(root, split, *, count=None, seed=0, include_mask=False):
    root = Path(root)
    data = load_sharded_split(root, split, count=count, seed=seed)
    indices = np.asarray(data["indices"])
    n = len(indices)
    logits = np.empty((n, 9 * 24 * 24), np.float32)
    sampled = np.empty((n, 5), np.int32)
    masks = np.empty((n, 24, 24, 4), bool) if include_mask else None
    facts, captions = [], []
    offset = destination = 0
    for shard in json.loads((root / "manifest.json").read_text())[split]["shards"]:
        idx = indices[(indices >= offset) & (indices < offset + shard["samples"])] - offset
        if len(idx):
            with np.load(root / shard["path"] / "samples.npz", allow_pickle=False) as arrays:
                teacher = arrays["logits"][idx]
                obs = arrays["observations"][idx]
                actions = arrays["actions"][idx]
                if include_mask:
                    masks[destination:destination + len(idx)] = arrays["masks"][idx]
            end = destination + len(idx)
            logits[destination:end], sampled[destination:end] = teacher, actions
            batch_facts = [decision_facts(o, a) for o, a in zip(obs, greedy_actions(teacher))]
            facts.extend(batch_facts)
            captions.extend(decision_caption(f) for f in batch_facts)
            destination = end
        offset += shard["samples"]
    data.update(logits=logits, sampled=sampled, decision_facts=facts,
                grounding_captions=data["captions"], captions=captions)
    if include_mask:
        data["hidden"] = pack_legal_masks(data["hidden"], masks)
    return data
