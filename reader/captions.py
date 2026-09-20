"""Observable-fact warm-start targets, never claimed to be policy rationales."""
import numpy as np


def observation_facts(obs):
    """Only player-visible augmented inputs are accepted, never simulator state."""
    obs = np.asarray(obs, dtype=np.float32)
    own_army, enemy_army = float(obs[17, 0, 0]), float(obs[19, 0, 0])
    if own_army > enemy_army * 1.1:
        balance = "Own army is larger."
    elif enemy_army > own_army * 1.1:
        balance = "Enemy army is larger."
    else:
        balance = "Armies are similar in size."
    visibility = "Enemy troops are visible." if np.any(obs[11] > 0) else "No enemy troops are visible."
    # Channel 6 remembers discovered generals; 10 and 11 are current ownership.
    general_known = np.any((obs[6] > 0) & (obs[10] == 0))
    general = "Enemy general has been located." if general_known else "Enemy general has not been located."
    own_land, enemy_land = float(obs[16, 0, 0]), float(obs[18, 0, 0])
    if own_land > enemy_land:
        land = "Own territory is larger."
    elif enemy_land > own_land:
        land = "Enemy territory is larger."
    else:
        land = "Territory sizes are equal."
    return " ".join([balance, visibility, general, land])
