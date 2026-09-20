import numpy as np
from reader.decisions import decision_caption, decode_action
from reader.decision_audit import matched_decision_pairs, decision_probability_audit


def caption(action):
    return decision_caption({**decode_action(action, 2), "destination": "neutral", "general_relation": "closer"})


def test_tie_score_and_probability_mass_are_kept_separate():
    logits = np.full((2, 36), -1e9)
    logits[:, 0:2] = 2
    logits[:, 32:] = 1
    result = decision_probability_audit([caption(1), "not an action"], logits)
    assert result["exact_action_accuracy"] == 0
    assert result["top_score_set_accuracy"] == .5
    assert result["parse_rate"] == .5
    assert result["mean_logit_regret_on_parseable"] == 0


def test_matched_pairs_require_same_map_and_facts_but_changed_decision():
    logits = np.zeros((4, 36))
    logits[np.arange(4), [0, 1, 1, 0]] = 1
    data = {"logits": logits, "rows": [{"map_id": m} for m in ["a", "a", "b", "b"]],
            "facts": [{"general": {x}} for x in ["left", "left", "left", "right"]]}
    assert matched_decision_pairs(data) == [(0, 1)]
