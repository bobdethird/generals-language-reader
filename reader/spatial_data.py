"""Load sharded activation data without keeping full observations in memory."""
import json
from pathlib import Path
import numpy as np
from reader.behavior import grounding_caption
from reader.grounding import expected_facts


def spatial_targets(observations):
    """Soft-label masks in patch-row/patch-col/local-row/local-col order."""
    observations = np.asarray(observations)
    if observations.shape[1:] != (38, 24, 24):
        raise ValueError("Spatial supervision requires 38x24x24 observations")
    own = observations[:, 1]
    enemy = observations[:, 2]
    general = (observations[:, 6] > 0) & (observations[:, 10] > 0)
    own_max, enemy_max = own.max((1, 2)), enemy.max((1, 2))
    fields = np.stack([(own == own_max[:, None, None]) & (own_max[:, None, None] > 0),
                       general,
                       (enemy == enemy_max[:, None, None]) & (enemy_max[:, None, None] > 0)], 1)
    patches = fields.reshape(len(fields), 3, 8, 3, 8, 3).transpose(0, 1, 2, 4, 3, 5)
    own_total, enemy_total = observations[:, 17, 0, 0], observations[:, 19, 0, 0]
    # 0 = own larger, 1 = enemy larger, 2 = similar.
    balance = np.where(own_total > enemy_total * 1.1, 0,
                       np.where(enemy_total > own_total * 1.1, 1, 2)).astype(np.int64)
    return patches.reshape(len(fields), 3, 576), balance


def load_sharded_split(root, split, *, count=None, seed=0):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    shards = manifest[split]["shards"]
    total = sum(s["samples"] for s in shards)
    # Evaluation selects at most one position per map before adding any repeats.
    all_rows = []
    for shard in shards:
        rows = [json.loads(line) for line in (root / shard["path"] / "captions.jsonl").read_text().splitlines()]
        if len(rows) != shard["samples"]:
            raise ValueError("Shard row count mismatch")
        all_rows.extend(rows)
    if len(all_rows) != total:
        raise ValueError("Dataset size mismatch")
    if count is None:
        selected = np.arange(total)
    else:
        order = np.random.default_rng(seed).permutation(total)
        first, rest, seen = [], [], set()
        for i in order:
            group = all_rows[i]["map_id"]
            if group not in seen:
                first.append(i)
                seen.add(group)
            else:
                rest.append(i)
        selected = np.sort(np.asarray((first + rest)[:count]))
    n = len(selected)
    hidden = np.empty((n, 67, 448), dtype=np.float32)
    spatial = np.empty((n, 3, 576), dtype=bool)
    balance = np.empty(n, dtype=np.int64)
    captions, facts, rows = [], [], []
    offset = destination = 0
    for shard in shards:
        idx = selected[(selected >= offset) & (selected < offset + shard["samples"])] - offset
        if len(idx):
            with np.load(root / shard["path"] / "samples.npz", allow_pickle=False) as arrays:
                h = arrays["activations"][idx]
                observations = arrays["observations"][idx]
            if h.dtype != np.float32:
                raise ValueError("Dataset must preserve the original float32 activations")
            end = destination + len(idx)
            hidden[destination:end] = h
            spatial[destination:end], balance[destination:end] = spatial_targets(observations)
            captions.extend(grounding_caption(o) for o in observations)
            facts.extend(expected_facts(o) for o in observations)
            # Treat repeated episodes on the same map as one bootstrap group.
            rows.extend({**all_rows[int(offset + i)], "episode_id": all_rows[int(offset + i)]["game_id"],
                         "game_id": all_rows[int(offset + i)]["map_id"]} for i in idx)
            destination = end
        offset += shard["samples"]
    return {"hidden": hidden, "spatial": spatial, "balance": balance,
            "captions": captions, "facts": facts, "rows": rows, "indices": selected.tolist()}
