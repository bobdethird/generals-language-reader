import numpy as np
from reader.behavior import grounding_caption
from reader.grounding import (FIELDS, expected_facts, parse_facts, score_facts,
                              summarize_facts, cross_game_shuffle, grounding_gate)


def board():
    obs = np.zeros((38, 24, 24), dtype=np.float32)
    obs[1, 20, 1] = 50
    obs[6, 2, 20] = obs[10, 2, 20] = 1
    obs[2, 12, 12] = 30
    obs[17] = 100
    obs[19] = 50
    return obs


def test_automatic_facts_round_trip_and_visible_locations():
    obs = board()
    assert all(score_facts(grounding_caption(obs), expected_facts(obs)).values())
    assert expected_facts(obs)["enemy_region"] == {"middle center"}
    obs[2] = 0
    obs[20, 1, 1] = 1000  # Stale memory must not become a visible enemy.
    assert expected_facts(obs)["enemy_region"] == {"none"}
    assert all(score_facts(grounding_caption(obs), expected_facts(obs)).values())


def test_ties_are_accepted_but_wrong_or_contradictory_claims_are_not():
    obs = board()
    obs[1, 2, 2] = 50
    facts = expected_facts(obs)
    assert score_facts("Largest friendly force is in the upper left.", facts)["friendly_region"]
    assert not score_facts("Largest friendly force is in the lower right.", facts)["friendly_region"]
    assert not score_facts("Our general is not in the upper right.", facts)["general_region"]
    assert not score_facts("Own army is larger. Enemy army is larger.", facts)["army_balance"]
    assert not score_facts("Own army is larger. Own army is not larger.", facts)["army_balance"]
    assert not score_facts("No enemy troops are visible. Largest visible enemy force is in the middle center.", facts)["enemy_region"]


def test_parser_handles_simple_region_synonyms_and_missing_claims():
    assert parse_facts("Our general is in the top middle.")["general_region"] == {"upper center"}
    facts = expected_facts(board())
    metrics, _ = summarize_facts(["The game is in progress."], [facts])
    assert metrics["mean_fact_accuracy"] == 0
    assert metrics["recognized_claim_coverage"] == 0


def test_shuffle_always_uses_another_game_and_gate_requires_all_fields():
    rows = [{"game_id": str(i // 3)} for i in range(12)]
    for i, j in enumerate(cross_game_shuffle(rows)):
        assert rows[i]["game_id"] != rows[j]["game_id"]
    metrics = {"per_field_accuracy": dict.fromkeys(FIELDS, .9), "visible_enemy_location_accuracy": .9}
    interval = {"accuracy_gain": .25, "game_bootstrap_95_percent_interval": [.1, .4]}
    assert grounding_gate(metrics, interval)
    metrics["per_field_accuracy"]["general_region"] = .2
    assert not grounding_gate(metrics, interval)
