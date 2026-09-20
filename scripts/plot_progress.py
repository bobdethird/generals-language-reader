"""Graph a read-only training snapshot with curriculum-aware evaluation curves."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re

from plot_training import read_run, trailing_mean, plt, np, METRIC, EVAL

STAGE = re.compile(r"CURRICULUM stage (\d+)/\d+ \(iter (\d+),.*dist=(\d+)-(\d+)")
MATCH = re.compile(r"(_current|_ema) vs (iter_\d+): (\d+)W/(\d+)L/(\d+)D")
REF_END = re.compile(r"REF_EVAL ELO \(iter (\d+)\)")


def parse_progress(path, config):
    offset = config.get("iteration_offset", 0)
    local_iteration, stage = 0, 0
    transitions, random_evaluations, references, pending = [], [], [], []
    for line in path.read_text().splitlines():
        match = METRIC.search(line)
        if match:
            local_iteration = int(match[1])
        match = STAGE.search(line)
        if match:
            stage, step, lo, hi = map(int, match.groups())
            assert step == local_iteration
            transitions.append(dict(iteration=offset + step, stage=stage, distance=[lo, hi]))
        match = EVAL.search(line)
        if match:
            wins, losses, draws = map(int, match.groups())
            random_evaluations.append(dict(iteration=offset + local_iteration, stage=stage,
                                           wins=wins, losses=losses, draws=draws,
                                           win_rate=wins / (wins + losses + draws)))
        match = MATCH.search(line)
        if match:
            candidate, opponent = match.groups()[:2]
            wins, losses, draws = map(int, match.groups()[2:])
            games = wins + losses + draws
            assert games == 2 * config["ref_eval_games"]
            pending.append(dict(candidate=candidate, opponent=opponent, wins=wins, losses=losses,
                                draws=draws, games=games, win_rate=wins / games))
        match = REF_END.search(line)
        if match:
            # Evaluation occurs before the printed upcoming iteration, including iteration 1.
            assert int(match[1]) - 1 == local_iteration
            assert len(pending) == 4
            references.extend(dict(iteration=offset + local_iteration, **row) for row in pending)
            pending = []
    return transitions, random_evaluations, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    history = [read_run(root / "runs" / name) for name in ("modal-h100-pilot-v1", "modal-h100-continue-v1")]
    current = read_run(args.snapshot, "client-output.log")
    runs = history + [current]
    for previous, following in zip(runs, runs[1:]):
        assert following["rows"][0]["iteration"] == previous["rows"][-1]["iteration"] + 1
        assert following["record"]["resume"]["source_run"] == previous["path"].name
    cfg = current["config"]
    transitions, random_evals, references = parse_progress(args.snapshot / "client-output.log", cfg)
    # Independently validate streamed counts against persisted structured evaluations.
    persisted = [json.loads(line) for line in (args.snapshot / "reference-evaluations/evaluations.jsonl").read_text().splitlines()]
    steps = sorted(set(row["iteration"] for row in references))
    for evaluation in persisted:
        step = steps[evaluation["evaluation_index"]]
        for row in [row for row in references if row["iteration"] == step]:
            expected = evaluation["h2h"][row["candidate"]][row["opponent"]]
            assert all(row[key] == expected[key] for key in ("wins", "losses", "draws"))
    rows = [row for run in runs for row in run["rows"]]
    last = rows[-1]["iteration"]
    target = cfg["iteration_offset"] + cfg["num_iters"]
    samples = sum(len(run["rows"]) * 2 * run["config"]["num_envs"] * run["config"]["num_steps"] for run in runs)
    snapshot = json.loads((args.snapshot / "snapshot.json").read_text())
    captured = snapshot["live_console_capture_at"]
    metadata = dict(cumulative_iteration=last, target_iteration=target, percent_complete=100 * last / target,
                    player_samples=samples, snapshot_captured_at=captured,
                    app_status=snapshot["observed_app_status"], curriculum_transitions=transitions,
                    random_evaluations=random_evals, reference_evaluations=references,
                    source_paths=[str(run["path"].resolve()) for run in runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in ((".csv", rows), ("-reference.csv", references)):
        path = args.output.parent / (args.output.name + suffix)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    blue, orange, gray = "#176B96", "#BD5C27", "#A9B8C2"
    stage_colors = ["#8A9AA6", "#6AABBD", "#267F94", "#143F5C"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#172D3C",
                         "axes.labelcolor": "#172D3C", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#CFD9DE",
                         "xtick.color": "#536774", "ytick.color": "#536774", "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 2, figsize=(14, 12.5))
    fig.subplots_adjust(left=.085, right=.97, bottom=.13, top=.84, hspace=.62, wspace=.26)
    fig.suptitle("Generals.io: training progress", x=.085, y=.971, ha="left", fontsize=22, fontweight="bold")
    fig.text(.085, .935, f"{last:,} / {target:,} iterations ({100 * last / target:.2f}%)   ·   "
             f"{samples / 1e6:,.1f}M player samples   ·   One H100", fontsize=12, color="#536774")
    stamp = dt.datetime.fromisoformat(captured).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(.085, .902, f"Snapshot: {stamp}. Training was running when checked.", fontsize=10, color="#536774")
    fig.text(.085, .873, "Loss curves: faint = each iteration; bold = trailing 25-iteration mean. Dashed lines mark resumed runs.",
             fontsize=9.5, color="#536774")

    ax = axes[0, 0]
    for opponent, color in (("iter_0300", blue), ("iter_1500", orange)):
        data = [row for row in references if row["candidate"] == "_ema" and row["opponent"] == opponent]
        ax.plot([row["iteration"] for row in data], [100 * row["win_rate"] for row in data],
                color=color, marker="o", markersize=4, lw=2, label=f"vs iteration {int(opponent[5:]):,}")
        tail = data[-1]
        ax.annotate(f"{tail['win_rate']:.1%}", (tail["iteration"], tail["win_rate"] * 100),
                    xytext=(5, -4), textcoords="offset points", color=color, fontsize=10, fontweight="bold")
    ax.set_title("EMA wins against frozen checkpoints", loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylabel("Wins / all 128 games (%)")
    ax.set_xlim(cfg["iteration_offset"] - 30, last + 180)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    for ax, key, title, ylabel in [
        (axes[0, 1], "total_loss", "Total training loss", "Loss (unitless)"),
        (axes[1, 0], "value_loss", "Value prediction loss", "Cross-entropy (nats)"),
        (axes[1, 1], "policy_loss", "Policy loss", "PPO surrogate loss (unitless)")]:
        for run in runs:
            x = [row["iteration"] for row in run["rows"]]
            y = [row[key] for row in run["rows"]]
            ax.plot(x, y, color=gray, lw=.65, alpha=.6)
            ax.plot(x, trailing_mean(y, 25), color=blue, lw=1.8)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, last * 1.025)
        for boundary in (300, 1500):
            ax.axvline(boundary, color="#81919D", ls=(0, (4, 4)), lw=.8)
        if key == "policy_loss":
            ax.axhline(0, color="#81919D", lw=.6)

    ax = axes[2, 0]
    for run in history:
        data = [ev for ev in run["evaluations"] if ev["kind"] == "current policy, periodic"]
        ax.plot([ev["iteration"] for ev in data], [100 * ev["win_rate"] for ev in data],
                color=stage_colors[0], marker="o", markersize=2.5, lw=1.2)
    for stage, config in enumerate(cfg["curriculum"]):
        data = [ev for ev in random_evals if ev["stage"] == stage]
        ax.plot([ev["iteration"] for ev in data], [100 * ev["win_rate"] for ev in data],
                color=stage_colors[stage], marker="o", markersize=3, lw=1.5,
                label=f"Distance {config['min_generals_distance']}–{config['max_generals_distance']}")
    ax.axhline(60, color="#81919D", ls=":", lw=.9)
    ax.set_title("Random-opponent wins, by difficulty", loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylabel("Wins / all evaluation games (%)")
    ax.set_xlim(0, last * 1.025)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    ax.text(last * .03, 62, "60% promotion threshold", fontsize=8.5, color="#536774")

    ax = axes[2, 1]
    stage_x = [cfg["iteration_offset"]] + [row["iteration"] for row in transitions] + [last]
    stage_y = [0] + [row["stage"] for row in transitions] + [transitions[-1]["stage"] if transitions else 0]
    ax.step(stage_x, stage_y, where="post", color=blue, lw=2)
    ax.scatter([row["iteration"] for row in transitions], [row["stage"] for row in transitions], color=blue, s=20)
    ax.set_yticks(range(len(cfg["curriculum"])), [f"{s}: {c['min_generals_distance']}–{c['max_generals_distance']}" for s, c in enumerate(cfg["curriculum"])])
    ax.set_ylabel("Stage: general-distance range (steps)")
    ax.set_title("Curriculum advancement", loc="left", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlim(cfg["iteration_offset"] - 50, last * 1.025)
    ax.set_ylim(-.2, len(cfg["curriculum"]) - .6)

    for ax in axes.flat:
        ax.set_xlabel("Cumulative PPO iteration")
        ax.grid(axis="y", color="#DFE6EB", lw=.6)
        ax.set_axisbelow(True)
    fig.text(.085, .081,
             "Source: original H100 pilot + continuation + live curriculum run (local snapshot; no training changes).\n"
             "Reference tests: 64 paired maps / 128 games per opponent, fixed environment; draws remain in the win-rate denominator.\n"
             "Random tests: 64 games before iteration 1,500; 128 afterward. Difficulty changes are shown separately; loss alone is not playing strength.",
             fontsize=8.7, color="#536774", va="top", linespacing=1.6)
    for suffix in (".png", ".svg"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"plot":str(args.output.with_suffix('.png').resolve()),
                      "iteration":last,"percent_complete":100*last/target,
                      "current_stage":stage_y[-1],"latest_reference_tests":references[-4:],
                      "snapshot":stamp},indent=2))


if __name__ == "__main__":
    main()
