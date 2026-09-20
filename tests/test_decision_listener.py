import numpy as np
import torch
from reader.decision_listener import context_features, DecisionListener


def test_listener_context_cannot_read_explicit_action_fields():
    suffix = " The destination is neutral territory. This moves farther from our general."
    texts = ["Send half the troops from row 3 column 4 north." + suffix,
             "Send all but one troop from row 20 column 15 west." + suffix,
             suffix]
    features = context_features(texts)
    np.testing.assert_array_equal(features[0], features[1])
    np.testing.assert_array_equal(features[0], features[2])
    assert features[0].tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert not context_features(["Send half the troops from row 3 column 4 north."]).any()


def test_listener_rejects_contradictory_semantic_codes():
    text = "The destination is friendly territory. The destination is neutral territory."
    assert not context_features([text]).any()


def test_listener_has_no_actor_input_and_masks_illegal_moves():
    model = DecisionListener(width=48, layers=1)
    board, history = torch.randn(2, 38, 24, 24), torch.randn(2, 2, 512)
    legal = torch.ones(2, 4609, dtype=torch.bool)
    legal[:, 8] = False
    output = model(board, history, torch.zeros(2, 8), legal)
    assert output.shape == (2, 4609) and (output[:, 8] == -1e9).all()
    other = model(board, history, torch.ones(2, 8), legal)
    assert not torch.allclose(output[:, -1], other[:, -1])
