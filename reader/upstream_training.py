"""Narrow runtime fixes around pinned AverageJoe curriculum and evaluation.

The self-play collector, rewards, network and PPO objective remain upstream code.
"""
import inspect
import json
from pathlib import Path


def patched_train_source(source):
    """Invalidate only the map generator and initial-state cache at stage changes."""
    replacements = {
        "                # Regenerate pool with new params (pool generator recompiles, rollout does NOT)":
        "                # The simulator caches self as static; mutated distances need a new trace.\n"
        "                p_init_envs = _refresh_curriculum_generation(env, _init_envs)\n"
        "                # Regenerate pool with new params (pool generator recompiles, rollout does NOT)",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Pinned upstream curriculum changed; review compatibility patch")
        source = source.replace(before, after)
    return source


def refresh_curriculum_generation(env, initializer):
    import jax
    env._make_pool_batch.clear_cache()
    # A fresh function identity is required: pmap(initializer) can reuse its old trace.
    return jax.pmap(lambda key: initializer(key))


def make_cached_match():
    """Reuse upstream's JIT match loop across checkpoints of the same structure."""
    import evals.matchup as matchup
    source = inspect.getsource(matchup.play_match)
    ending = "    w_a, w_b, d = _play(agent_a.network, agent_b.network, key)\n    return int(w_a), int(w_b), int(d)"
    if source.count(ending) != 1:
        raise RuntimeError("Pinned upstream match loop changed; review executor cache")
    source = source.replace("def play_match(", "def make_executor(", 1)
    source = source.replace(ending, "    return _play")
    namespace = dict(vars(matchup))
    exec(compile(source, "<AverageJoe cached match factory>", "exec"), namespace)
    factory = namespace["make_executor"]
    cache = {}

    def play(agent_a, agent_b, env, pool, num_games, truncation, key):
        signature = (env, id(pool), num_games, truncation, agent_a.pad_to, agent_b.pad_to,
                     agent_a.greedy_fn, agent_b.greedy_fn, agent_a.augment_fn,
                     agent_b.augment_fn, agent_a.init_obs_state_fn)
        if signature not in cache:
            cache[signature] = factory(agent_a, agent_b, env, pool, num_games, truncation, key)
        scores = cache[signature](agent_a.network, agent_b.network, key)
        result = tuple(map(int, scores))
        if sum(result) != num_games:
            raise ValueError("Reference match did not account for every game")
        return result

    play.executors = cache
    return play


def paired_match(play, agent_a, agent_b, env, pool, games_per_side, truncation, key):
    """Both player positions on identical maps; count outcomes from A's view."""
    forward = play(agent_a, agent_b, env, pool, games_per_side, truncation, key)
    reverse = play(agent_b, agent_a, env, pool, games_per_side, truncation, key)
    result = dict(wins=forward[0] + reverse[1], losses=forward[1] + reverse[0],
                  draws=forward[2] + reverse[2])
    if sum(result.values()) != 2 * games_per_side:
        raise ValueError("Paired evaluation game count mismatch")
    return result


def make_reference_evaluator(play, artifact_dir):
    import jax.random as jrandom
    from evals.matchup import compute_elo, merge_h2h

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    call_index = 0

    def evaluate(candidates, ref_agents, ref_h2h, env, pool, num_games, truncation, key):
        nonlocal call_index
        if ref_h2h is None:
            raise ValueError("Reference matrix must be a dictionary")
        # Build the fixed reference-vs-reference matrix once in the SAME environment.
        if not ref_h2h:
            for i, first in enumerate(ref_agents):
                ref_h2h.setdefault(first.name, {})
                for second in ref_agents[i + 1:]:
                    key, match_key = jrandom.split(key)
                    result = paired_match(play, first, second, env, pool, num_games, truncation, match_key)
                    ref_h2h[first.name][second.name] = result
                    ref_h2h.setdefault(second.name, {})[first.name] = dict(
                        wins=result["losses"], losses=result["wins"], draws=result["draws"])
                    print(f"REFERENCE BASELINE {first.name} vs {second.name}: {result}", flush=True)
            (artifact_dir / "reference-matrix.json").write_text(
                json.dumps({"h2h": ref_h2h, "games_per_side": num_games}, indent=2) + "\n")
        h2h = {agent.name: {} for agent in candidates + ref_agents}
        for candidate in candidates:
            for ref in ref_agents:
                key, match_key = jrandom.split(key)
                result = paired_match(play, candidate, ref, env, pool, num_games, truncation, match_key)
                h2h[candidate.name][ref.name] = result
                h2h[ref.name][candidate.name] = dict(wins=result["losses"],
                                                    losses=result["wins"], draws=result["draws"])
                print(f"  {candidate.name} vs {ref.name}: {result['wins']}W/{result['losses']}L/"
                      f"{result['draws']}D ({result['wins'] / (2 * num_games):.1%} overall; both seats)", flush=True)
        full_h2h = merge_h2h(ref_h2h, h2h)
        names = [agent.name for agent in candidates + ref_agents]
        ratings = compute_elo(names, full_h2h)
        with (artifact_dir / "evaluations.jsonl").open("a") as file:
            file.write(json.dumps(dict(evaluation_index=call_index, games_per_side=num_games,
                                       ratings=ratings, h2h=full_h2h)) + "\n")
        call_index += 1
        return ratings, full_h2h

    return evaluate


def install_training_support(*, paired_references=True):
    """Opt-in compatibility patches and local structured metrics for curriculum runs."""
    import train.ppo as ppo
    import train.evaluations as evaluations
    from logger import Logger

    if getattr(ppo, "_project_curriculum_support", False):
        return
    source = patched_train_source(inspect.getsource(ppo.train))
    ppo._refresh_curriculum_generation = refresh_curriculum_generation
    exec(compile(source, "<AverageJoe curriculum cache compatibility>", "exec"), vars(ppo))
    if paired_references:
        evaluations.ref_eval = make_reference_evaluator(make_cached_match(), "reference-evaluations")
    original_log = Logger.log

    def log(self, step, metrics):
        with Path("metrics.jsonl").open("a") as file:
            file.write(json.dumps(dict(step=int(step), **{k: float(v) for k, v in metrics.items()})) + "\n")
        return original_log(self, step, metrics)

    Logger.log = log
    ppo._project_curriculum_support = True
    print(f"Enabled curriculum cache refresh and JSONL metrics; paired references={paired_references}.", flush=True)
