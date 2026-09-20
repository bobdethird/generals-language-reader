"""Plot recorded PPO losses and evaluations, including checkpoint continuations.

Uses local artifacts only; this does not start training or allocate a GPU.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "generals-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "generals-plot-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ruamel.yaml import YAML

METRIC = re.compile(
    r"Iter\s+(\d+)/(\d+)\s+\| Loss: ([^|]+)\| PG: ([^|]+)\| VF: ([^|]+)\| Ent: ([^|]+)\|"
)
EVAL = re.compile(r"^\s*EVAL: (\d+)W/(\d+)L/(\d+)D")
METRICS = ("total_loss", "policy_loss", "value_loss", "entropy")


def read_run(path, log_name="training.log"):
    config = YAML(typ="safe").load((path / "requested-config.yaml").read_text())
    record = json.loads((path / "run.json").read_text())
    offset = config.get("iteration_offset", 0)
    rows, evaluations = [], []
    completed = 0
    for line in (path / log_name).read_text().splitlines():
        match = METRIC.search(line)
        if match:
            iteration, target = map(int, match.groups()[:2])
            if iteration != completed + 1 or target != config["num_iters"]:
                raise ValueError(f"Missing or inconsistent iterations in {path}")
            completed = iteration
            values = list(map(float, match.groups()[2:]))
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metrics in {path}, iteration {iteration}")
            rows.append(dict(run=path.name, local_iteration=iteration,
                             iteration=offset + iteration, **dict(zip(METRICS, values))))
        evaluation = EVAL.search(line)
        if evaluation:
            wins, losses, draws = map(int, evaluation.groups())
            games = wins + losses + draws
            if games != config["eval_games"]:
                raise ValueError(f"Unexpected evaluation size in {path}")
            # Upstream evaluates BEFORE its next update (including at iteration zero).
            evaluations.append(dict(run=path.name, iteration=offset + completed,
                                    wins=wins, losses=losses, draws=draws,
                                    win_rate=wins / games, games=games,
                                    kind="current policy, periodic"))
    if not rows:
        raise ValueError(f"No loss metrics in {path}")
    if record.get("status") == "completed" and completed != config["num_iters"]:
        raise ValueError(f"Completed run has incomplete logs: {path}")
    heldout_path = path / "heldout-evaluation.json"
    if heldout_path.exists():
        heldout = json.loads(heldout_path.read_text())
        relative = Path(heldout["checkpoint"]).relative_to("/runs")
        checkpoint = path.parent / relative
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != heldout["checkpoint_sha256"]:
            raise ValueError(f"Evaluation/checkpoint hash mismatch in {path}")
        evaluations.append(dict(run=path.name, iteration=offset + completed,
                                kind="EMA checkpoint, held-out", **heldout))
    return dict(path=path, config=config, record=record, rows=rows, evaluations=evaluations)


def trailing_mean(values, window):
    sums = np.r_[0.0, np.cumsum(values, dtype=float)]
    right = np.arange(1, len(values) + 1)
    left = np.maximum(0, right - window)
    return (sums[right] - sums[left]) / (right - left)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="Output file stem")
    parser.add_argument("--smooth", type=int, default=25, help="Trailing mean window per run")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be positive")
    runs = [read_run(path) for path in args.runs]
    runs.sort(key=lambda run: run["rows"][0]["iteration"])
    for previous, current in zip(runs, runs[1:]):
        if current["rows"][0]["iteration"] != previous["rows"][-1]["iteration"] + 1:
            raise ValueError("Runs must form a contiguous training sequence")
        resume = current["record"].get("resume", {})
        if resume.get("source_run") != previous["path"].name:
            raise ValueError("Continuation provenance does not match preceding run")
    rows = [row for run in runs for row in run["rows"]]
    evaluations = [evaluation for run in runs for evaluation in run["evaluations"]]
    samples = sum(len(run["rows"]) * 2 * run["config"]["num_envs"] *
                  run["config"]["num_steps"] * run["record"].get("gpu_count", 1) for run in runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(iterations=len(rows), player_samples=samples,
                   smoothing=f"Trailing {args.smooth}-iteration mean, reset at each run boundary",
                   sources=[str(run["path"].resolve()) for run in runs],
                   statuses=[run["record"].get("status") for run in runs],
                   evaluations=evaluations)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    blue, orange, gray, text_color = "#176B96", "#BD5C27", "#A9B8C2", "#172D3C"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": text_color, "text.color": text_color,
                         "xtick.color": "#536774", "ytick.color": "#536774",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#CED7DD", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))
    fig.subplots_adjust(left=.075, right=.96, bottom=.17, top=.82, hspace=.48, wspace=.24)
    fig.suptitle("Generals.io training: loss and playing performance", x=.075, y=.965,
                 ha="left", fontsize=20, fontweight="bold")
    all_complete = all(run["record"].get("status") == "completed" for run in runs)
    fig.text(.075, .921,
             f"{len(rows):,} PPO iterations  /  {samples / 1e6:,.1f} million player samples  /  "
             f"{'Completed' if all_complete else 'Snapshot'}  /  {runs[-1]['record']['gpu_requested']}",
             fontsize=11, color="#536774")
    fig.text(.075, .878, "Faint lines: recorded loss. Bold lines: "
             f"trailing {args.smooth}-iteration mean within each run. Dashed line: checkpoint resume.",
             fontsize=10, color="#536774")
    titles = [("total_loss", "Total training loss", "Loss (unitless)"),
              ("value_loss", "Value prediction loss", "Cross-entropy (nats)"),
              ("policy_loss", "Policy loss", "PPO surrogate loss (unitless)")]
    for ax, (key, title, ylabel) in zip(axes.flat, titles):
        for run in runs:
            x = [row["iteration"] for row in run["rows"]]
            y = [row[key] for row in run["rows"]]
            ax.plot(x, y, color=gray, alpha=.6, lw=.65)
            ax.plot(x, trailing_mean(y, args.smooth), color=blue, lw=2)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel)
        if key == "policy_loss":
            ax.axhline(0, color="#81919D", lw=.7, alpha=.6)

    ax = axes[1, 1]
    for run in runs:
        periodic = [ev for ev in run["evaluations"] if ev["kind"] == "current policy, periodic"]
        ax.plot([ev["iteration"] for ev in periodic], [100 * ev["win_rate"] for ev in periodic],
                color=blue, marker="o", markersize=3, lw=1.5)
    heldout = [ev for ev in evaluations if ev["kind"] == "EMA checkpoint, held-out"]
    for ev in heldout:
        ax.scatter(ev["iteration"], 100 * ev["win_rate"], s=55, marker="D", color=orange,
                   edgecolors="white", linewidths=.8, zorder=5)
        ax.annotate(f"{100 * ev['win_rate']:.1f}%", (ev["iteration"], 100 * ev["win_rate"]),
                    xytext=(-8, -18), textcoords="offset points", ha="right",
                    fontsize=10, fontweight="bold", color=orange)
    ax.set_title("Win rate against a random opponent", loc="left", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Wins / all games (%)")
    ax.set_ylim(0, 108)
    ax.legend(handles=[Line2D([0], [0], color=blue, marker="o", markersize=3,
                             label="Current policy · periodic, 64 games"),
                       Line2D([0], [0], color=orange, marker="D", linestyle="none",
                              markersize=5, label="EMA checkpoint · held-out, 128 games")],
              loc="lower right", fontsize=8.5, frameon=False)
    boundaries = [run["config"].get("iteration_offset", 0) for run in runs[1:]]
    last_iteration = rows[-1]["iteration"]
    for ax in axes.flat:
        ax.set_xlim(0, last_iteration * 1.025)
        ax.set_xlabel("Cumulative PPO iteration")
        ax.grid(axis="y", color="#DFE6EB", lw=.6)
        ax.set_axisbelow(True)
        for boundary in boundaries:
            ax.axvline(boundary, ls=(0, (4, 4)), color="#788A97", lw=1, zorder=0)
    if boundaries:
        axes[0, 0].text(boundaries[0] + last_iteration * .012, .95,
                         f"Resume at {boundaries[0]}", transform=axes[0, 0].get_xaxis_transform(),
                         fontsize=9, color="#536774", va="top")
    date = runs[-1]["record"]["started_at"][:10]
    fig.text(.075, .103, f"Source: {' + '.join(run['path'].name for run in runs)} · {date} UTC\n"
             "Losses are training-log values (4 decimal places). Held-out EMA evaluations use the same 64 paired maps.\n"
             "Self-play losses need not decrease as play improves. Random-opponent wins do not measure strength against skilled players.",
             fontsize=9, color="#536774", linespacing=1.6, va="top")
    for extension in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"plot": str(args.output.with_suffix('.png').resolve()),
                      "iterations": len(rows), "player_samples": samples,
                      "heldout": [{k: ev[k] for k in ('iteration', 'wins', 'losses', 'draws', 'win_rate')}
                                  for ev in heldout]}, indent=2))


if __name__ == "__main__":
    main()
