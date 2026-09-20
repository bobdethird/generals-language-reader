"""Auditable scoring of four visible facts in generated English.

These checks measure specific factual claims, not explanations or strategy.
Unrecognized wording is counted as missing, never assumed correct.
"""
import re
import numpy as np

FIELDS = ("friendly_region", "general_region", "enemy_region", "army_balance")


def region(row, col, height, width):
    return (("upper", "middle", "lower")[min(2, 3 * row // height)] + " " +
            ("left", "center", "right")[min(2, 3 * col // width)])


def expected_facts(obs):
    h, w = obs.shape[-2:]

    def largest(channel):
        maximum = obs[channel].max()
        if maximum <= 0:
            return {"none"}
        # Any tied largest force is a valid answer.
        return {region(r, c, h, w) for r, c in np.argwhere(obs[channel] == maximum)}

    general = {region(r, c, h, w) for r, c in np.argwhere((obs[6] > 0) & (obs[10] > 0))}
    own, enemy = float(obs[17, 0, 0]), float(obs[19, 0, 0])
    balance = "own" if own > enemy * 1.1 else "enemy" if enemy > own * 1.1 else "similar"
    return {"friendly_region": largest(1), "general_region": general or {"unknown"},
            "enemy_region": largest(2), "army_balance": {balance}}


def parse_facts(text):
    claims = {field: set() for field in FIELDS}
    for sentence in re.split(r"[.!?;\n]", text.lower()):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        if re.search(r"(?:no|not any) (?:visible )?enemy (?:troops|forces|armies)", sentence):
            claims["enemy_region"].add("none")
        field = None
        if re.search(r"(?:our|own|friendly) general", sentence):
            field = "general_region"
        elif re.search(r"largest (?:visible )?(?:friendly|own) (?:force|army|troop)", sentence):
            field = "friendly_region"
        elif re.search(r"largest (?:visible )?enemy (?:force|army|troop)", sentence):
            field = "enemy_region"
        if field:
            matches = re.findall(r"\b(upper|top|middle|central|lower|bottom)[ -]+(left|center|middle|right)\b", sentence)
            for vertical, horizontal in matches:
                vertical = {"top": "upper", "bottom": "lower", "central": "middle"}.get(vertical, vertical)
                horizontal = "center" if horizontal == "middle" else horizontal
                claims[field].add("negated" if re.search(r"\bnot\b", sentence) else f"{vertical} {horizontal}")
        if re.search(r"\b(?:own|our|friendly) army is larger\b", sentence):
            claims["army_balance"].add("own")
        if re.search(r"\benemy army is larger\b", sentence):
            claims["army_balance"].add("enemy")
        if re.search(r"\barmies are (?:similar|equal) in size\b", sentence):
            claims["army_balance"].add("similar")
        if re.search(r"\b(?:own|our|friendly|enemy) army is (?:not|never) larger\b", sentence):
            claims["army_balance"].add("negated")
    return claims


def score_facts(text, targets):
    claims = parse_facts(text)
    return {field: bool(claims[field]) and claims[field].issubset(targets[field]) for field in FIELDS}


def summarize_facts(texts, targets):
    scores = np.asarray([[score_facts(text, target)[field] for field in FIELDS]
                         for text, target in zip(texts, targets)], dtype=float)
    if not len(scores) or len(texts) != len(targets):
        raise ValueError("Text/target lengths must match and be nonempty")
    coverage = np.asarray([[bool(parse_facts(text)[field]) for field in FIELDS] for text in texts])
    visible = np.asarray([target["enemy_region"] != {"none"} for target in targets])
    metrics = {"positions": len(texts), "mean_fact_accuracy": float(scores.mean()),
               "all_four_correct": float(scores.all(axis=1).mean()),
               "recognized_claim_coverage": float(coverage.mean()),
               "per_field_accuracy": dict(zip(FIELDS, scores.mean(axis=0).tolist())),
               "visible_enemy_positions": int(visible.sum()),
               "visible_enemy_location_accuracy": float(scores[visible, 2].mean()) if visible.any() else None,
               "unique_descriptions": len(set(texts))}
    return metrics, scores


def cross_game_shuffle(rows, seed=718):
    rng = np.random.default_rng(seed)
    games = np.asarray([row["game_id"] for row in rows])
    result = []
    for game in games:
        choices = np.flatnonzero(games != game)
        if not len(choices):
            raise ValueError("Activation controls require at least two games")
        result.append(int(rng.choice(choices)))
    return np.asarray(result)


def paired_game_interval(real, control, rows, seed=442):
    differences = np.asarray(real).mean(axis=1) - np.asarray(control).mean(axis=1)
    grouped = {}
    for row, delta in zip(rows, differences):
        grouped.setdefault(row["game_id"], []).append(delta)
    groups = [np.asarray(values) for values in grouped.values()]
    rng = np.random.default_rng(seed)
    samples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
               for _ in range(2000)]
    return {"accuracy_gain": float(differences.mean()),
            "game_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "games": len(groups)}


def grounding_gate(metrics, comparison):
    """Predeclared prototype gate before experimenting with behavior RL."""
    return (min(metrics["per_field_accuracy"].values()) >= 0.80
            and (metrics["visible_enemy_location_accuracy"] or 0) >= 0.80
            and comparison["accuracy_gain"] >= 0.15
            and comparison["game_bootstrap_95_percent_interval"][0] > 0)
