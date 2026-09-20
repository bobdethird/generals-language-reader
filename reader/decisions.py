"""Automatic decision descriptions and independently parsed reconstruction.

Deployment uses argmax of the original 9-channel logits, then decodes all pass
encodings as one action. This differs from argmax AFTER merging pass probability.
The latter is appropriate for predicting a sampled action, not deployed greedy
play. Supervised captions describe observable mechanics, never inferred motives.
"""
import re
import numpy as np

DIRECTIONS = ("north", "south", "west", "east")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DECISION_PROMPT = (
    "Read the frozen generals.io player's internal activations. Describe its preferred "
    "move accurately, using rows and columns numbered from one. State whether it sends "
    "half or all but one of the source troops, the source row and column, and north, "
    "south, west, or east. Then describe the destination ownership and whether the "
    "move approaches our general. If it prefers to wait, say so. Describe supported "
    "facts only; do not invent strategic reasons or hidden enemies. Use three short sentences."
)


def greedy_actions(logits):
    logits = np.asarray(logits)
    cells = logits.shape[-1] // 9
    return np.minimum(logits.argmax(-1), 8 * cells)


def decode_action(index, size=24):
    index = int(index)
    if not 0 <= index <= 8 * size * size:
        raise ValueError("Invalid canonical action")
    if index == 8 * size * size:
        return {"pass": True}
    channel, cell = divmod(index, size * size)
    row, col = divmod(cell, size)
    return {"pass": False, "row": row, "col": col,
            "direction": DIRECTIONS[channel % 4], "half": channel >= 4}


def encode_action(action, size=24):
    if action["pass"]:
        return 8 * size * size
    row, col = action["row"], action["col"]
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError("Source outside board")
    channel = DIRECTIONS.index(action["direction"]) + 4 * int(action["half"])
    return channel * size * size + row * size + col


def decision_facts(observation, index):
    size = observation.shape[-1]
    action = decode_action(index, size)
    if action["pass"]:
        return {**action, "destination": "none", "general_relation": "none"}
    row, col = action["row"], action["col"]
    dr, dc = DELTAS[DIRECTIONS.index(action["direction"])]
    dest_row, dest_col = row + dr, col + dc
    if not (0 <= dest_row < size and 0 <= dest_col < size):
        raise ValueError("Preferred action leaves board; check action ordering/masks")
    if observation[10, dest_row, dest_col] > 0:
        owner = "friendly"
    elif observation[11, dest_row, dest_col] > 0:
        owner = "enemy"
    elif observation[12, dest_row, dest_col] > 0 or observation[13, dest_row, dest_col] > 0:
        owner = "unknown"
    elif observation[9, dest_row, dest_col] > 0:
        owner = "neutral"
    else:
        owner = "unknown"
    generals = np.argwhere((observation[6] > 0) & (observation[10] > 0))
    relation = "unknown"
    if len(generals) == 1:
        gr, gc = generals[0]
        delta = abs(dest_row - gr) + abs(dest_col - gc) - abs(row - gr) - abs(col - gc)
        relation = "closer" if delta < 0 else "farther" if delta > 0 else "same"
    return {**action, "destination": owner, "general_relation": relation}


def decision_caption(facts):
    if facts["pass"]:
        return "The preferred action is to wait. No troops move this turn."
    fraction = "half the troops" if facts["half"] else "all but one troop"
    action = (f"Send {fraction} from row {facts['row'] + 1} column {facts['col'] + 1} "
              f"{facts['direction']}.")
    owner = {"friendly": "The destination is friendly territory.",
             "enemy": "The destination is visible enemy territory.",
             "neutral": "The destination is neutral territory.",
             "unknown": "The destination ownership is unknown."}[facts["destination"]]
    relation = {"closer": "This moves closer to our general.",
                "farther": "This moves farther from our general.",
                "same": "This keeps the same distance from our general.",
                "unknown": "The direction relative to our general is unknown."}[facts["general_relation"]]
    return f"{action} {owner} {relation}"


MOVE = re.compile(r"send\s+(half the troops|all but one troop)\s+from row\s+(\d+)\s+column\s+(\d+)\s+(north|south|west|east)\b", re.I)


def parse_decision(text, size=24):
    """Text-only reconstruction; no board, hidden state, or teacher target input."""
    moves = list(MOVE.finditer(text))
    wait = bool(re.search(r"preferred action is to wait\b", text, re.I))
    if wait and not moves:
        return {"pass": True, "destination": "none", "general_relation": "none"}
    if len(moves) != 1 or wait:
        return None
    fraction, row, col, direction = moves[0].groups()
    action = {"pass": False, "row": int(row) - 1, "col": int(col) - 1,
              "direction": direction.lower(), "half": fraction.lower().startswith("half")}
    try:
        encode_action(action, size)
    except ValueError:
        return None
    owners = []
    for value, pattern in (("friendly", r"destination is friendly territory"),
                           ("enemy", r"destination is visible enemy territory"),
                           ("neutral", r"destination is neutral territory"),
                           ("unknown", r"destination ownership is unknown")):
        if re.search(pattern, text, re.I):
            owners.append(value)
    relations = []
    for value, pattern in (("closer", r"moves closer to our general"),
                           ("farther", r"moves farther from our general"),
                           ("same", r"keeps the same distance from our general"),
                           ("unknown", r"direction relative to our general is unknown")):
        if re.search(pattern, text, re.I):
            relations.append(value)
    return {**action, "destination": owners[0] if len(owners) == 1 else None,
            "general_relation": relations[0] if len(relations) == 1 else None}


def score_decisions(texts, targets, size=24):
    if len(texts) != len(targets) or not texts:
        raise ValueError("Mismatched or empty decision examples")
    parsed = [parse_decision(text, size) for text in texts]
    exact, complete, readable = [], [], []
    move_exact, fields = [], {key: [] for key in ("source", "direction", "half", "destination", "general_relation")}
    for prediction, target in zip(parsed, targets):
        readable.append(prediction is not None)
        correct = prediction is not None and encode_action(prediction, size) == encode_action(target, size)
        exact.append(correct)
        normalize = lambda s: re.sub(r"\s+", " ", s.strip().lower())
        supported = (prediction is not None and prediction["destination"] is not None
                     and prediction["general_relation"] is not None
                     and normalize(texts[len(exact) - 1]) == normalize(decision_caption(prediction)))
        complete.append(correct and supported and prediction["destination"] == target["destination"]
                        and prediction["general_relation"] == target["general_relation"])
        if not target["pass"]:
            move_exact.append(correct)
            moving = prediction is not None and not prediction["pass"]
            fields["source"].append(moving and (prediction["row"], prediction["col"]) == (target["row"], target["col"]))
            for field in ("direction", "half", "destination", "general_relation"):
                fields[field].append(moving and prediction[field] == target[field])
    return {"positions": len(texts), "move_positions": len(move_exact),
            "parse_rate": float(np.mean(readable)), "exact_action_accuracy": float(np.mean(exact)),
            "move_exact_accuracy": float(np.mean(move_exact)) if move_exact else None,
            "fully_correct_description": float(np.mean(complete)),
            "move_field_accuracy": {k: float(np.mean(v)) if v else None for k, v in fields.items()}}, np.asarray(exact)
