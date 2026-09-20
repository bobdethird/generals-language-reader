import numpy as np
import torch
from reader.decision_data import pack_legal_masks
from reader.decision_language import MaskedDecisionSpatialAdapter
from reader.language import ActivationAdapter


def test_mask_pack_respects_patch_cell_direction_order():
    hidden = np.zeros((1, 67, 448), np.float32)
    masks = np.zeros((1, 24, 24, 4), bool)
    row, col, direction = 8, 19, 3
    masks[0, row, col, direction] = True
    packed = pack_legal_masks(hidden, masks)
    patch = (row//3)*8 + col//3
    local = (row%3)*3 + col%3
    assert packed[0, patch+3, 448+local*4+direction] == 1
    assert packed[..., 448:].sum() == 1
    base = ActivationAdapter(448, 32)
    adapter = MaskedDecisionSpatialAdapter(base, 448, 32, width=32)
    x = torch.from_numpy(packed)
    features = adapter.policy_features(torch.zeros(1, 64, 81), x)
    assert features[0, patch, direction*9+local] > -1
    assert features[0, patch, (direction+4)*9+local] > -1
    assert (features[..., 8*9:] > -1).all()  # Original pass encodings remain legal.
    assert (features[..., :8*9] > -1).sum() == 2
    mapped, auxiliary = adapter.forward_with_aux(x)
    torch.testing.assert_close(mapped, base(x[..., :448]), atol=0, rtol=0)
    assert auxiliary["policy"].shape == (1, 5184)
