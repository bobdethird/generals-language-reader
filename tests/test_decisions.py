import numpy as np
import torch
from reader.decisions import (greedy_actions, decode_action, encode_action,
                              decision_facts, decision_caption, parse_decision, score_decisions)
from reader.decision_language import policy_distillation, DecisionSpatialAdapter
from reader.language import ActivationAdapter


def test_deployment_greedy_is_not_argmax_after_pass_aggregation():
    logits = np.full((1, 36), -100.)
    logits[0, 0] = 1.
    logits[0, 32:] = 0.
    assert greedy_actions(logits).tolist() == [0]
    assert np.exp(logits[0, 32:]).sum() > np.exp(logits[0, 0])


def test_all_action_encodings_roundtrip():
    for index in range(8 * 24 * 24 + 1):
        assert encode_action(decode_action(index)) == index


def test_decision_caption_roundtrip_and_unknown_destination():
    obs = np.zeros((38, 24, 24), np.float32)
    obs[6, 12, 12] = obs[10, 12, 12] = 1
    obs[12, 11, 10] = 1
    action = {"pass": False, "row": 12, "col": 10, "direction": "north", "half": True}
    facts = decision_facts(obs, encode_action(action))
    assert facts["destination"] == "unknown" and facts["general_relation"] == "farther"
    text = decision_caption(facts)
    assert "row 13 column 11" in text
    assert parse_decision(text) == facts
    metrics, scores = score_decisions([text], [facts])
    assert metrics["fully_correct_description"] == 1 and scores.all()
    changed = text.replace("north", "south")
    assert score_decisions([changed], [facts])[0]["exact_action_accuracy"] == 0
    extra = text + " It does this because it fears an attack."
    assert score_decisions([extra], [facts])[0]["fully_correct_description"] == 0
    assert parse_decision(text + " The preferred action is to wait.") is None


def test_unknown_is_not_a_claim_of_no_enemy():
    facts = {"pass": False, "row": 1, "col": 1, "direction": "east", "half": False,
             "destination": "unknown", "general_relation": "unknown"}
    text = decision_caption(facts)
    assert "unknown" in text and "no enemy" not in text
    assert score_decisions([""], [facts])[0]["parse_rate"] == 0
    assert parse_decision(text.replace("row 2", "row 99")) is None


def test_policy_distillation_finite_for_forced_wait_and_detached_teacher():
    student = torch.randn(2, 36, requires_grad=True)
    teacher = torch.randn(2, 36, requires_grad=True)
    with torch.no_grad():
        teacher[0, :32] = -1e9
    loss, metrics = policy_distillation(student, teacher)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert teacher.grad is None
    assert set(metrics) == {"policy_full_kl", "policy_move_kl", "policy_rank_loss"}


def test_exact_teacher_distribution_is_optimal_without_extra_confidence():
    teacher = torch.randn(4, 36)
    teacher[0, :32] = -1e9
    student = teacher.clone().requires_grad_(True)
    loss, metrics = policy_distillation(student, teacher)
    assert abs(float(loss.detach())) < 1e-6
    assert metrics["policy_rank_loss"] == 0


def test_policy_aux_adapter_starts_with_original_prefix():
    base = ActivationAdapter(16, 32)
    adapter = DecisionSpatialAdapter(base, 16, 32, width=32)
    hidden = torch.randn(2, 67, 16)
    mapped, auxiliary = adapter.forward_with_aux(hidden)
    torch.testing.assert_close(mapped, base(hidden), atol=0, rtol=0)
    assert auxiliary["policy"].shape == (2, 9 * 24 * 24)
    (mapped.square().mean() + auxiliary["policy"].square().mean()).backward()
    assert adapter.policy.weight.grad.abs().sum() > 0
