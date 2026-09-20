"""Decision fidelity diagnostics that do not hide passing or tied-action errors."""
import numpy as np
from reader.decisions import encode_action, parse_decision, greedy_actions


def matched_decision_pairs(data, limit=128, seed=812):
    """Same map and coarse board facts, different preferred moves.

    These are observational contrasts, not causal interventions. Selection uses
    only player targets and facts, never the reader's successes or failures.
    """
    preferred = greedy_actions(data["logits"])
    groups = {}
    for index, (row, facts) in enumerate(zip(data["rows"], data["facts"])):
        key = (row["map_id"], tuple((k, tuple(sorted(v))) for k, v in sorted(facts.items())))
        groups.setdefault(key, []).append(index)
    candidates = []
    for key, indices in groups.items():
        first = indices[0]
        alternatives = [i for i in indices[1:] if preferred[i] != preferred[first]]
        if alternatives:
            candidates.append((key[0], first, alternatives[0]))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected, seen_maps = [], set()
    for map_id, a, b in candidates:
        if map_id not in seen_maps:
            selected.append((a, b))
            seen_maps.add(map_id)
            if len(selected) == limit:
                break
    return selected


def decision_probability_audit(texts, logits):
    logits = np.asarray(logits, dtype=np.float64)
    if len(texts) != len(logits) or not len(texts):
        raise ValueError("Empty or mismatched audit data")
    cells = logits.shape[-1] // 9
    size = int(round(cells ** .5))
    if size * size != cells:
        raise ValueError("Non-square board")
    preferred = greedy_actions(logits)
    canonical_scores = np.concatenate([logits[:, :8*cells], logits[:, 8*cells:].max(-1, keepdims=True)], -1)
    # This merges MAX scores for greedy ties, not probability mass.
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    canonical_probabilities = np.concatenate([
        probabilities[:, :8*cells], probabilities[:, 8*cells:].sum(-1, keepdims=True)], -1)
    predicted = []
    for text in texts:
        decoded = parse_decision(text, size)
        predicted.append(encode_action(decoded, size) if decoded is not None else -1)
    predicted = np.asarray(predicted)
    valid = predicted >= 0
    exact = predicted == preferred
    tie_correct = np.zeros(len(logits), bool)
    gaps = np.full(len(logits), np.nan)
    gaps[valid] = canonical_scores[valid].max(-1) - canonical_scores[np.flatnonzero(valid), predicted[valid]]
    tie_correct[valid] = gaps[valid] <= 1e-6
    confidence = canonical_probabilities[np.arange(len(logits)), preferred]
    strata = []
    for lower, upper in ((0, .25), (.25, .5), (.5, .75), (.75, 1.000001)):
        selected = (confidence >= lower) & (confidence < upper)
        strata.append({"probability_range": [lower, min(upper, 1)], "positions": int(selected.sum()),
                       "exact_accuracy": float(exact[selected].mean()) if selected.any() else None})
    return {"positions": len(logits), "exact_action_accuracy": float(exact.mean()),
            "top_score_set_accuracy": float(tie_correct.mean()),
            "parse_rate": float(valid.mean()), "illegal_or_masked_predictions": int(np.sum(gaps > 1e8)),
            "mean_logit_regret_on_parseable": float(np.mean(gaps[valid])) if valid.any() else None,
            "confidence_strata": strata,
            "optimal_expected_sample_prediction_accuracy": float(canonical_probabilities.max(-1).mean()),
            "note": "Exact accuracy matches deployed tie-breaking. Top-score-set accuracy also accepts equal-score alternatives. "
                    "Confidence is the sampled policy probability of the deployed greedy action; these are distinct quantities."}
