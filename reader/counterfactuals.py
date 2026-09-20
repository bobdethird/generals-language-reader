"""Controlled input interventions for the frozen player's local decisions.

These are tests of the policy, not recovered thoughts or counterfactual game
outcomes. We preserve ownership, geometry, legal moves, and all past observations.
Current troop-count edits update the redundant current-observation channels,
the newest history delta, and the current public total together. No fogged tile
is edited. History ablation is explicitly a cue-removal test, not a possible game.
"""
import numpy as np
from reader.decisions import decode_action, DIRECTIONS, DELTAS

PROBES = (
    "reducing source troops", "increasing source troops",
    "reducing destination troops", "increasing destination troops",
    "reducing the nearest visible enemy force", "increasing the nearest visible enemy force",
    "reducing troops on our general", "increasing troops on our general",
    "removing recent friendly troop changes", "removing recent enemy troop changes",
)
FEATURES = 8
SCOPE = ("Local sensitivity of the frozen policy to controlled inputs, with its legal moves and "
         "past history held fixed. This does not establish a long-term intention or game outcome.")


def canonical_scores(logits):
    """Max, not probability sum, preserves deployed greedy pass semantics."""
    x = np.asarray(logits)
    cells = x.shape[-1] // 9
    return np.concatenate([x[..., :8*cells], x[..., 8*cells:].max(-1, keepdims=True)], -1)


def action_margin(logits, action):
    x = canonical_scores(logits).copy()
    chosen = x[..., action].copy()
    x[..., action] = -np.inf
    return chosen - x.max(-1)


def change_troops(observation, temporal, cell, amount):
    """Change current troops consistently while leaving past observations fixed."""
    o, t = np.array(observation, copy=True), np.array(temporal, copy=True)
    r, c = cell
    if o[12, r, c] or o[13, r, c] or o[8, r, c]:
        raise ValueError("Cannot intervene on hidden or impassable cells")
    old = float(o[0, r, c]); delta = amount - old
    friendly, enemy, neutral = (bool(o[k, r, c]) for k in (10, 11, 9))
    if sum((friendly, enemy, neutral)) != 1 or amount < 0:
        raise ValueError("Invalid visible troop edit")
    if friendly and amount < min(old, 2):
        raise ValueError("Troop intervention would change action legality")
    o[0, r, c] = amount
    for channel, owner in ((1, friendly), (2, enemy), (3, neutral)):
        o[channel, r, c] = amount if owner else 0
    if friendly:
        o[17] += delta
        o[24, r, c] += delta
    elif enemy:
        o[19] += delta
        o[31, r, c] += delta
        o[20, r, c] = amount
        t[0, -1] += delta
    return o, t


def make_interventions(observation, temporal, action):
    """Two doses per probe. Inactive probes are identity controls, never evidence."""
    o, t = np.asarray(observation), np.asarray(temporal)
    if o.shape != (38, 24, 24) or t.shape != (2, 512):
        raise ValueError("Expected published-player observation and temporal shapes")
    obs = np.broadcast_to(o, (len(PROBES), 2, *o.shape)).copy()
    history = np.broadcast_to(t, (len(PROBES), 2, *t.shape)).copy()
    active = np.zeros(len(PROBES), bool)
    metadata = [None] * len(PROBES)
    decoded = decode_action(action)
    generals = np.argwhere((o[6] > 0) & (o[10] > 0))
    source = target = None
    if not decoded["pass"]:
        source = (decoded["row"], decoded["col"])
        dr, dc = DELTAS[DIRECTIONS.index(decoded["direction"])]
        target = (source[0] + dr, source[1] + dc)
    general = tuple(generals[0]) if len(generals) == 1 else None
    enemies = np.argwhere((o[11] > 0) & (o[2] > 0) & (o[12] == 0) & (o[13] == 0))
    anchor = general or source
    enemy = None
    if len(enemies) and anchor is not None:
        enemy = tuple(enemies[np.abs(enemies - np.array(anchor)).sum(-1).argmin()])
    for group, cell in enumerate((source, target, enemy, general)):
        if cell is None or not all(0 <= v < 24 for v in cell):
            continue
        r, c = cell
        if o[12, r, c] or o[13, r, c] or o[8, r, c] or not any(o[k, r, c] for k in (9, 10, 11)):
            continue
        old = float(o[0, r, c])
        # Empty neutral land cannot suddenly contain an army. Zero-army owned
        # cells are left unchanged too, avoiding changes to legal action masks.
        if old <= 0:
            continue
        floor = 2 if o[10, r, c] and old >= 2 else 1
        for up in (0, 1):
            amounts = ([max(floor, np.floor(old * f)) for f in (.75, .5)] if not up
                       else [np.ceil(old * f) for f in (1.5, 2)])
            if amounts[0] == old or amounts[1] == old:
                continue
            # A friendly cell with one troop must remain unable to move.
            if o[10, r, c] and old < 2 and any(a >= 2 for a in amounts):
                continue
            k = 2 * group + up
            for dose, amount in enumerate(amounts):
                obs[k, dose], history[k, dose] = change_troops(o, t, cell, amount)
            active[k] = True
            metadata[k] = {"row": int(r), "col": int(c), "original": old,
                           "changed": list(map(float, amounts)), "kind": "troop_count"}
    for k, channels in ((8, slice(24, 31)), (9, slice(31, 38))):
        if np.any(o[channels]):
            # Two removal strengths guard against a single arbitrary ablation.
            obs[k, 0, channels] *= .5
            obs[k, 1, channels] = 0
            active[k] = True
            metadata[k] = {"kind": "history_cue_ablation", "remaining_fraction": [.5, 0.]}
    return obs, history, active, metadata


def summarize_effects(original_logits, modified_logits, active, threshold=.15,
                      *, identity_logits=None, archived_logits=None):
    """Measured features only; selected rationale is a training target, not input."""
    baseline = canonical_scores(original_logits)
    action = int(baseline.argmax())
    baseline_margin = float(action_margin(original_logits, action))
    controls = [np.asarray(original_logits)]
    if identity_logits is not None:
        controls.extend(np.asarray(identity_logits).reshape(-1,len(original_logits)))
    if archived_logits is not None:
        controls.append(np.asarray(archived_logits))
    controls = np.stack(controls)
    stable = bool(np.all(canonical_scores(controls).argmax(-1)==action))
    # Two logits enter a margin. A conservative multiple of the largest null
    # drift prevents low-precision replays being presented as causal effects.
    logit_drift = float(np.max(np.abs(controls-np.asarray(original_logits))))
    margin_drift = float(np.max(np.abs(action_margin(controls,action)-baseline_margin)))
    noise_bound = max(margin_drift,2*logit_drift)
    effective_threshold = threshold + 4*noise_bound
    modified = np.asarray(modified_logits)
    changes = action_margin(modified, action) - baseline_margin
    switched = canonical_scores(modified).argmax(-1) != action
    # Two-dose agreement and non-trivial effects are required for a claim.
    consistent = ((changes[:, 0] * changes[:, 1] > 0)
                  & (np.abs(changes).min(-1) >= effective_threshold) & active & stable)
    # Smooth compression preserves relative strength even for large effects;
    # hard clipping would make distinct strongest-probe targets indistinguishable.
    features = np.stack([active, np.arcsinh(changes[:, 0])/3,
                         np.arcsinh(changes[:, 1])/3, switched[:, 0], switched[:, 1],
                         consistent, np.full(len(active),np.log1p(max(baseline_margin,0))/3),
                         np.full(len(active),stable)], -1).astype(np.float32)
    strength = np.where(consistent, np.abs(changes).min(-1), -1)
    k = int(strength.argmax()) if consistent.any() else -1
    rationale = {"probe": k, "effect": "none" if k < 0 else "weakens" if changes[k, 1] < 0 else "strengthens",
                 "switches": bool(switched[k, 1]) if k >= 0 else False}
    return features, rationale, {"baseline_action": action, "baseline_margin": baseline_margin,
            "margin_changes": changes.tolist(), "switched": switched.tolist(), "consistent": consistent.tolist(),
            "numerically_stable_action":stable,"null_logit_drift":logit_drift,
            "null_margin_drift":margin_drift,"effective_threshold":effective_threshold}


def rationale_caption(rationale):
    k = rationale["probe"]
    if k < 0:
        return "The tested changes do not identify a consistent influence on this choice."
    first = f"In controlled tests, {PROBES[k]} {rationale['effect']} its preference for the original action."
    second = ("The stronger test changes its preferred action." if rationale["switches"]
              else "It still prefers the original action in the stronger test.")
    return first + " " + second


def parse_rationale(text):
    """Strictly reject invented or contradictory causal claims."""
    normalized = " ".join(text.strip().lower().split())
    for k in range(-1, len(PROBES)):
        for effect in (("none",) if k < 0 else ("weakens", "strengthens")):
            for switches in ((False,) if k < 0 or effect == "strengthens" else (False, True)):
                candidate = {"probe": k, "effect": effect, "switches": switches}
                if normalized == rationale_caption(candidate).lower():
                    return candidate
    return None


def verify_rationale(text,audit):
    """Verify actual claims, separately from matching an arbitrary tie break.

Two probes can name the same physical tile (e.g. source = our general). A true
explanation about either tied strongest influence is supported. Canonical-target
accuracy remains a separate, stricter training/evaluation metric.
"""
    prediction=parse_rationale(text)
    if prediction is None:return {"claims_supported":False,"strongest_influence":False}
    consistent=np.asarray(audit["consistent"],bool)
    if prediction["probe"]<0:
        correct=not consistent.any()
        return {"claims_supported":correct,"strongest_influence":correct}
    k=prediction["probe"]
    changes=np.asarray(audit["margin_changes"])
    sign="weakens" if changes[k,1]<0 else "strengthens"
    supported=bool(consistent[k] and prediction["effect"]==sign
                   and prediction["switches"]==audit["switched"][k][1])
    strength=np.where(consistent,np.abs(changes).min(-1),-1)
    strongest=supported and bool(np.isclose(strength[k],strength.max(),atol=1e-6,rtol=0))
    return {"claims_supported":supported,"strongest_influence":strongest}
