"""Project paths and narrow compatibility support for pinned upstream sources."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def import_upstream():
    path = str(ROOT / "vendor/averagejoe")
    if path not in sys.path:
        sys.path.insert(0, path)


def install_jax_compat():
    """Restore the removed pmap helper using JAX's documented replacement.

    https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements-for-device-put-sharded-and-device-put-replicated
    This only controls array placement; it does not change PPO or observations.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    if hasattr(jax, "device_put_replicated"):
        return

    def replicate(tree, devices):
        sharding = NamedSharding(Mesh(np.array(devices), ("x",)), PartitionSpec("x"))
        return jax.tree.map(
            lambda value: jax.device_put(jnp.stack([value] * len(devices)), sharding), tree
        )

    jax.device_put_replicated = replicate


def install_simulator_compat():
    """Average Joe uses the simulator's former name for castles (cities)."""
    from generals.core.env import GeneralsEnv

    if not hasattr(GeneralsEnv, "num_cities_range"):
        GeneralsEnv.num_cities_range = property(
            lambda self: self.num_castles_range,
            lambda self, value: setattr(self, "num_castles_range", value),
        )
