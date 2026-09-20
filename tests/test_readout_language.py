import numpy as np
import torch
from reader.language import ActivationAdapter
from reader.readout_language import ReadoutSpatialAdapter
from reader.decision_data import pack_legal_masks


def test_frozen_readout_respects_spatial_order_mask_and_split():
    hidden = np.zeros((2, 67, 448), np.float32)
    masks = np.zeros((2, 24, 24, 4), bool)
    masks[0, 8, 19, 3] = True
    adapter = ReadoutSpatialAdapter(ActivationAdapter(448, 32), 448, 32, width=32)
    # Give half-east a larger preference than full-east and wait.
    adapter.head_bias[7*9:8*9] = 2
    x = torch.from_numpy(pack_legal_masks(hidden, masks))
    logits = adapter.read_logits(x)
    assert logits[0].argmax().item() == 7*576 + 8*24 + 19
    assert logits[1].argmax().item() >= 4608
    assert logits[0, 3*576 + 8*24 + 19] == 0
    assert (logits[0, :4608] > -1e8).sum() == 2
    mapped, aux = adapter.forward_with_aux(x)
    assert mapped.shape == (2, 73, 32)
    assert aux['readout_policy'].shape == (2, 5184)
    mapped.square().mean().backward()
    assert adapter.row_words.weight.grad is not None
    assert adapter.head_weight.grad is None and adapter.head_bias.grad is None
    assert 'head_weight' not in dict(adapter.named_parameters())


def test_readout_rounding_matches_b200_bf16_bias_addition():
    adapter = ReadoutSpatialAdapter(ActivationAdapter(448, 32), 448, 32, width=32)
    with torch.no_grad():
        adapter.head_weight[0, 0] = 1
        adapter.head_bias[0] = .005859375
    hidden = np.zeros((1, 67, 448), np.float32)
    hidden[:, 3:, 0] = 1
    x = torch.from_numpy(pack_legal_masks(hidden, np.ones((1, 24, 24, 4), bool)))
    scores = adapter.read_logits(x)
    assert scores[0, 0] == 1.0078125
    assert scores[0, 0] != 1.005859375  # A float32 bias addition is a different policy.
