import numpy as np
import torch
from reader.behavior import (canonical_targets, canonical_actions, legal_actions,
                             behavior_scores, distillation_loss, eos_mask,
                             group_advantages, grounding_caption, MovePredictor)


def test_pass_equivalence_preserves_mass_and_action_identity():
    logits = torch.randn(3, 9 * 4)
    target = canonical_targets(logits)
    original = logits.softmax(-1)
    torch.testing.assert_close(target[:, :-1], original[:, :32])
    torch.testing.assert_close(target[:, -1], original[:, 32:].sum(-1))
    assert canonical_actions([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], 2).tolist() == [32, 32]


def test_masks_match_channel_major_full_then_half_actions():
    masks = np.zeros((1, 2, 2, 4), dtype=bool)
    masks[0, 1, 0, 3] = True
    legal = legal_actions(masks)
    assert torch.where(legal[0])[0].tolist() == [14, 30, 32]


def test_no_legal_moves_is_finite_and_not_movement_accuracy():
    logits = torch.tensor([[-1e9, -1e9, 0.0], [0.0, 2.0, 1.0]], requires_grad=True)
    targets = logits.detach().softmax(-1)
    scores = behavior_scores(logits, targets)
    assert scores['has_moves'].tolist() == [False, True]
    torch.testing.assert_close(scores['full_kl'], torch.zeros(2), atol=1e-6, rtol=0)
    loss = distillation_loss(logits, targets)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_rewards_compare_only_candidates_for_the_same_position():
    rewards = torch.tensor([1., 3., 100., 100.])
    torch.testing.assert_close(group_advantages(rewards, 2), torch.tensor([-2., 2., 0., 0.]))
    assert eos_mask(torch.tensor([[5, 2, 2], [5, 6, 2]]), 2).tolist() == [[True, True, False], [True, True, True]]


def test_grounding_uses_visible_enemy_not_remembered_armies():
    obs = np.zeros((38, 9, 9), dtype=np.float32)
    obs[1, 7, 1] = 50
    obs[20, 1, 1] = 100  # stale memory is not a currently visible enemy
    text = grounding_caption(obs)
    assert 'lower left' in text
    assert 'No enemy troops are visible.' in text


def test_predictor_masks_invalid_moves_and_text_can_affect_output():
    torch.manual_seed(3)
    model = MovePredictor(16)
    obs, temporal = torch.randn(2, 38, 9, 9), torch.randn(2, 2, 512)
    legal = torch.ones(2, 649, dtype=torch.bool)
    legal[:, 10] = False
    first = model(obs, temporal, torch.zeros(2, 16), legal)
    second = model(obs, temporal, torch.ones(2, 16), legal)
    assert (first[:, 10] == -1e9).all()
    assert not torch.allclose(first[:, -1], second[:, -1])
