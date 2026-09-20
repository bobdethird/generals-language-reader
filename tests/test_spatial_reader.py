import copy
import numpy as np
import torch
from reader.language import ActivationAdapter
from reader.spatial import SpatialAdapter, coordinate_features, factual_token_weights
from reader.spatial_data import spatial_targets


def test_layout_coordinates_distinguish_rows_columns_and_opposite_edges():
    xy = coordinate_features()
    assert xy.shape == (64, 18)
    assert len(torch.unique(xy, dim=0)) == 64
    assert xy[0, 0] == xy[7, 0] and xy[0, 1] == xy[56, 1]
    assert not torch.allclose(xy[0], xy[56])
    assert not torch.allclose(xy[0], xy[7])


def test_spatial_residual_preserves_initial_reader_and_has_gradients():
    torch.manual_seed(4)
    base = ActivationAdapter(16, 32)
    spatial = SpatialAdapter(copy.deepcopy(base), 16, 32, width=32)
    h = torch.randn(2, 67, 16)
    output, aux = spatial.forward_with_aux(h)
    torch.testing.assert_close(output, base(h), atol=0, rtol=0)
    assert aux["locations"].shape == (2, 3, 576)
    assert aux["balance"].shape == (2, 3)
    (output.square().mean() + aux["locations"].square().mean()).backward()
    assert spatial.output.weight.grad.abs().sum() > 0
    assert spatial.position.weight.grad.abs().sum() > 0
    assert h.grad is None


def test_spatial_targets_keep_exact_tile_order_ties_and_absent_enemy():
    obs = np.zeros((2, 38, 24, 24), dtype=np.float32)
    obs[:, 6, 8, 19] = obs[:, 10, 8, 19] = 1
    obs[:, 1, 5, 2] = obs[:, 1, 21, 16] = 20
    obs[0, 2, 12, 12] = 10
    obs[:, 17] = 40
    obs[:, 19] = 20
    targets, balance = spatial_targets(obs)
    index = ((8 // 3) * 8 + 19 // 3) * 9 + (8 % 3) * 3 + 19 % 3
    assert targets[:, 1, index].all()
    assert targets[:, 0].sum(1).tolist() == [2, 2]
    assert targets[:, 2].sum(1).tolist() == [1, 0]
    assert balance.tolist() == [0, 0]


def test_only_factual_value_tokens_get_extra_weight():
    caption = "Our general is in the upper right."
    offsets = torch.tensor([[[0, 3], [4, 11], [22, 27], [28, 33], [0, 0]]])
    weights = factual_token_weights([caption], offsets, torch.tensor([[1, 1, 1, 1, 0]]), 5)
    assert weights.tolist() == [[1, 1, 5, 5, 0]]
