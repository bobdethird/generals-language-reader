import json
import numpy as np
from reader.spatial_data import load_sharded_split


def test_sharded_loader_preserves_float32_and_groups_repeated_maps(tmp_path):
    shards = []
    for shard in range(2):
        folder = tmp_path / f"part-{shard}"
        folder.mkdir()
        hidden = np.full((3, 67, 448), np.float32(.18881875276565552 + shard))
        observations = np.zeros((3, 38, 24, 24), dtype=np.float32)
        observations[:, 1, 8, 19] = 30
        observations[:, 6, 8, 19] = observations[:, 10, 8, 19] = 1
        observations[:, 17] = 30
        np.savez(folder / "samples.npz", activations=hidden, observations=observations)
        rows = [{"game_id": f"episode-{shard}-{i}", "map_id": f"map-{shard}", "index": i}
                for i in range(3)]
        (folder / "captions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        shards.append({"path": folder.name, "samples": 3})
    (tmp_path / "manifest.json").write_text(json.dumps({"validation": {"shards": shards}}))
    data = load_sharded_split(tmp_path, "validation", count=2, seed=3)
    assert data["hidden"].dtype == np.float32
    np.testing.assert_array_equal(data["hidden"][:, 0, 0],
                                  np.array([.18881875276565552, 1.18881875276565552], np.float32))
    assert {row["game_id"] for row in data["rows"]} == {"map-0", "map-1"}
    assert all(row["episode_id"].startswith("episode-") for row in data["rows"])
    assert data["spatial"].shape == (2, 3, 576)
    assert data["facts"][0]["general_region"] == {"middle right"}
