"""Compare saved GPU pilots or explicit batch-scaling tests; never launches jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
from ruamel.yaml import YAML


GPU_USD_PER_SECOND = {"H100": 0.001097, "H200": 0.001261,
                      "B200": 0.001736, "B300": 0.001972}
MAX_HOST_USD_PER_SECOND = 2 * 0.0000131 + 16 * 0.00000222
ITERATION = re.compile(r"Iter\s+(\d+)/(\d+).*?SPS: (\d+)")
PHASES = re.compile(r"Iter\s+(\d+)/\d+.*?\(rollout ([\d.]+)s, ppo ([\d.]+)s\)")
SCALING_FIELDS = {"run_name", "num_envs", "num_steps", "minibatch_size", "num_iters", "ema_decay",
                  "eval_every", "ckpt_every", "save_every"}


def summarize(path):
    run = json.loads((path / "run.json").read_text())
    config = (path / "requested-config.yaml").read_bytes()
    assert hashlib.sha256(config).hexdigest() == run["config_sha256"]
    cfg = YAML(typ="safe").load(config)
    gpu_count = run.get("gpu_count", 1)
    if len(run["jax_devices"]) != gpu_count:
        raise ValueError(f"Device count does not match request: {path}")
    iterations = [(int(i), int(total), int(sps)) for i, total, sps in
                  ITERATION.findall((path / "training.log").read_text())]
    if not iterations or run["status"] != "completed":
        raise ValueError(f"Incomplete pilot: {path}")
    expected = iterations[-1][1]
    if expected != cfg["num_iters"]:
        raise ValueError(f"Iteration count differs from configuration: {path}")
    if [i for i, _, _ in iterations] != list(range(1, expected + 1)):
        raise ValueError(f"Missing or duplicated training iterations: {path}")
    # Same initial ten iterations excluded for every GPU; retain map-pool resets.
    warm_sps = [sps for i, _, sps in iterations if i > 10]
    if not warm_sps or min(warm_sps) <= 0:
        raise ValueError(f"Insufficient positive throughput measurements: {path}")
    evaluation = json.loads((path / "heldout-evaluation.json").read_text())
    checkpoint = path / "checkpoints" / cfg["run_name"] / f"{cfg['run_name']}_ema_{expected}.eqx"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == evaluation["checkpoint_sha256"]
    rate = gpu_count * GPU_USD_PER_SECOND[run["gpu_requested"]]
    warm_rate = statistics.harmonic_mean(warm_sps)
    phases = [(float(rollout), float(ppo)) for i, rollout, ppo in
              PHASES.findall((path / "training.log").read_text()) if int(i) > 10]
    local_samples = 2 * cfg["num_envs"] * cfg["num_steps"]
    minibatches = int(local_samples * cfg["adv_top_frac"]) // cfg["minibatch_size"]
    return {
        "run": str(path.resolve()), "gpu": run["gpu_requested"],
        "gpu_count": gpu_count, "configuration": cfg,
        "parallel_games": gpu_count * cfg["num_envs"],
        "global_optimizer_batch": gpu_count * cfg["minibatch_size"],
        "optimizer_updates": expected * cfg["num_epochs"] * minibatches,
        "training_player_samples": expected * local_samples * gpu_count,
        "hardware": run["nvidia_smi"], "config_sha256": run["config_sha256"],
        "upstream": run["upstream"], "iterations": expected,
        "compilation_cache": run.get("compilation_cache", "shared volume cache; prior baseline"),
        "training_seconds": run["seconds"], "total_seconds": run["wall_seconds"],
        "warm_iteration_range": [11, expected],
        "warm_player_samples_per_second": warm_rate,
        "median_iteration_player_samples_per_second": statistics.median(warm_sps),
        "mean_warm_rollout_seconds_rounded_logs": statistics.mean(x[0] for x in phases),
        "mean_warm_ppo_seconds_rounded_logs": statistics.mean(x[1] for x in phases),
        "gpu_usd_per_hour": rate * 3600,
        "estimated_gpu_cost_usd": rate * run["wall_seconds"],
        "estimated_cost_with_max_host_usd": (rate + MAX_HOST_USD_PER_SECOND) * run["wall_seconds"],
        "warm_player_samples_per_gpu_dollar": warm_rate / rate,
        "evaluation": {k: evaluation[k] for k in ("games", "wins", "losses", "draws", "win_rate", "seed")},
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scaling", action="store_true",
                        help="Allow declared batch/parallelism changes; require equal total samples")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = [summarize(path) for path in args.runs]
    if not args.scaling and len({row["config_sha256"] for row in runs}) != 1:
        raise ValueError("Training configurations differ; this is not a matched comparison")
    if args.scaling:
        fixed = [{k: v for k, v in row["configuration"].items() if k not in SCALING_FIELDS}
                 for row in runs]
        if any(cfg != fixed[0] for cfg in fixed):
            raise ValueError("Scaling runs differ in model, game, or other training settings")
        if len({row["training_player_samples"] for row in runs}) != 1:
            raise ValueError("Scaling runs must use equal total player samples")
    if any(row["upstream"] != runs[0]["upstream"] for row in runs):
        raise ValueError("Upstream code revisions differ")
    result = {
        "pricing_date": "2026-09-19", "pricing_source": "https://modal.com/pricing",
        "comparison": "batch and device scaling" if args.scaling else "matched GPU pilots",
        "method": "One pilot per configuration. Harmonic mean of logged SPS after the first ten iterations, including periodic map-pool resets. SPS counts both players; divide by two for environment transitions.",
        "limitations": [
            "Warm iteration timing includes rollout and PPO, but excludes evaluation and checkpoint saving. Total seconds includes those plus compilation and initialization.",
            "Total seconds excludes Modal image building, container startup, volume commit and local downloads. Cost estimates are not actual bills.",
            "The original H100 pilot used a shared compilation cache; new runs use fresh temporary caches. Prefer warm throughput for hardware comparison.",
            "This small model and batch may underuse larger GPUs; results do not predict full-size Average Joe training speed.",
            "One seed per configuration; numeric differences can change self-play trajectories. These evaluations do not establish a reliable policy-quality ranking.",
        ], "runs": runs,
    }
    if args.scaling:
        result["limitations"].extend([
            "Equal samples does not mean equivalent learning: larger optimizer batches reduce optimizer updates, and more parallel games change policy-refresh frequency and entropy scheduling per sample.",
            "Wide, wide-batch, and dual use EMA decay 0.99^4 to preserve its sample-based averaging horizon; their evaluation and checkpoint cadence differ from the original pilot. Wide-short preserves the baseline schedules.",
            "The dual run averages gradients for one policy; top-advantage filtering is performed separately on each device, so it is not numerically identical to the single-device run.",
            "The wide-short variant preserves the baseline collection batch, optimizer batch, update count, and schedules by collecting 64 turns from each of 1,024 games; its shorter GAE horizon and different game sample diversity can still affect learning.",
        ])
    for row in runs:
        row["warm_speedup_over_first"] = row["warm_player_samples_per_second"] / runs[0]["warm_player_samples_per_second"]
        row["warm_samples_per_dollar_ratio_over_first"] = row["warm_player_samples_per_gpu_dollar"] / runs[0]["warm_player_samples_per_gpu_dollar"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
