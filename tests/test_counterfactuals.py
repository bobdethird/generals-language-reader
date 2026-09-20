import numpy as np
import pytest
from reader.counterfactuals import (canonical_scores, make_interventions, change_troops,
    summarize_effects, rationale_caption, parse_rationale, PROBES)


def state():
    o = np.zeros((38, 24, 24), np.float32)
    o[10, 4, 4] = o[6, 4, 4] = 1
    o[0, 4, 4] = o[1, 4, 4] = 20
    o[11, 4, 5] = 1
    o[0, 4, 5] = o[2, 4, 5] = o[20, 4, 5] = 8
    o[17] = 20; o[19] = 30
    o[24, 4, 4] = 3; o[31, 4, 5] = -2
    t = np.zeros((2, 512), np.float32); t[0, -1] = 30
    return o, t


def test_edits_preserve_previous_state_and_update_duplicate_channels():
    o, t = state()
    x, h = change_troops(o, t, (4, 5), 16)
    assert x[0, 4, 5] == x[2, 4, 5] == x[20, 4, 5] == 16
    assert np.all(x[19] == 38) and h[0, -1] == 38
    assert x[31, 4, 5] == 6
    assert x[2, 4, 5] - x[31, 4, 5] == o[2, 4, 5] - o[31, 4, 5]
    np.testing.assert_array_equal(h[:, :-1], t[:, :-1])
    assert o[0, 4, 5] == 8 and t[0, -1] == 30
    o[12, 4, 5] = 1
    with pytest.raises(ValueError): change_troops(o, t, (4, 5), 16)


def test_probes_preserve_legality_geometry_and_exclude_hidden_cells():
    o, t = state(); action = 3*576 + 4*24 + 4
    x, h, valid, meta = make_interventions(o, t, action)
    assert valid.all() and x.shape == (10, 2, 38, 24, 24)
    for k in range(10):
        for dose in range(2):
            np.testing.assert_array_equal(x[k, dose, 6:14], o[6:14])
            np.testing.assert_array_equal(x[k, dose, 0] > 1, o[0] > 1)
    o[12, 4, 5] = 1
    _, _, valid, _ = make_interventions(o, t, action)
    assert not valid[2:6].any()


def test_max_pass_semantics_and_two_dose_evidence():
    logits = np.full(5184, -10., np.float32); logits[10] = 2; logits[20] = 1
    logits[4608:] = 0
    assert canonical_scores(logits).argmax() == 10
    changed = np.broadcast_to(logits, (10, 2, 5184)).copy()
    changed[0, 0, 10] = 1.5; changed[0, 1, 10] = .5
    changed[1, 0, 10] = 3; changed[1, 1, 10] = 1
    active = np.ones(10, bool)
    features, target, audit = summarize_effects(logits, changed, active)
    assert target == {"probe": 0, "effect": "weakens", "switches": True}
    assert not audit["consistent"][1]
    assert features.shape == (10, 8)
    assert parse_rationale(rationale_caption(target)) == target
    assert parse_rationale(rationale_caption(target) + " It wants to defend.") is None
    active[:] = False
    _, target, _ = summarize_effects(logits, changed, active)
    assert target["probe"] == -1


def test_wait_and_one_troop_cannot_gain_new_legal_moves():
    o, t = state(); o[0, 4, 4] = o[1, 4, 4] = 1
    _, _, active, _ = make_interventions(o, t, 4608)
    assert not active[:4].any() and not active[6:8].any()


def test_rounding_drift_is_not_a_causal_explanation():
    logits=np.full(5184,-10.,np.float32);logits[10]=2;logits[20]=1
    modified=np.broadcast_to(logits,(10,2,5184)).copy()
    modified[0,:,10]=[1.8,1.7]
    null=logits.copy();null[20]=1.125
    _,r,a=summarize_effects(logits,modified,np.ones(10,bool),identity_logits=null[None])
    assert r["probe"]==-1 and a["effective_threshold"]>1
    # Strong, repeatable changes still qualify when they exceed the null bound.
    modified[0,:,10]=[.5,0.]
    _,r,a=summarize_effects(logits,modified,np.ones(10,bool),identity_logits=null[None])
    assert r["probe"]==0 and a["numerically_stable_action"]
    null[20]=3
    _,r,a=summarize_effects(logits,modified,np.ones(10,bool),identity_logits=null[None])
    assert r["probe"]==-1 and not a["numerically_stable_action"]


def test_faithfulness_distinguishes_valid_alternative_from_strongest():
    from reader.counterfactuals import verify_rationale
    logits=np.full(5184,-10.,np.float32);logits[10]=2;logits[20]=1
    modified=np.broadcast_to(logits,(10,2,5184)).copy()
    modified[0,:,10]=[-1,-2];modified[6,:,10]=[-1,-2];modified[2,:,10]=[1.5,1.4]
    _,target,audit=summarize_effects(logits,modified,np.ones(10,bool))
    assert target["probe"]==0
    tied={"probe":6,"effect":"weakens","switches":True}
    assert verify_rationale(rationale_caption(tied),audit)=={"claims_supported":True,"strongest_influence":True}
    other={"probe":2,"effect":"weakens","switches":False}
    assert verify_rationale(rationale_caption(other),audit)=={"claims_supported":True,"strongest_influence":False}
    other["switches"]=True
    assert not verify_rationale(rationale_caption(other),audit)["claims_supported"]
