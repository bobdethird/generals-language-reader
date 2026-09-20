"""Plot individual-fact and fully correct-description validation accuracy."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/generals-reader-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for line in args.metrics.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("stage") == "validation":
            records[row["epoch"]] = row
    if not records:
        raise ValueError("No completed validation records")
    rows = [records[key] for key in sorted(records)]
    epochs = [row["epoch"] for row in rows]
    minutes = [row["seconds"] / 60 for row in rows]
    facts = [row["metrics"]["real"]["mean_fact_accuracy"] for row in rows]
    full = [row["metrics"]["real"]["all_four_correct"] for row in rows]
    selected = rows[-1]["selected_best_epoch"]
    selected_i = epochs.index(selected)
    count = rows[-1]["metrics"]["real"]["positions"]
    games = rows[-1]["comparisons"]["real_vs_shuffled"]["games"]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "accuracy.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["additional_pass", "elapsed_minutes", "individual_fact_accuracy", "all_four_correct"])
        writer.writerows(zip(epochs, minutes, facts, full))

    ink, muted, grid = "#183747", "#627c89", "#e4ebee"
    teal, blue = "#098878", "#2875a0"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12.8, 8.3), facecolor="white")
    fig.subplots_adjust(left=.10, right=.95, bottom=.24, top=.71)
    fig.text(.10, .94, "Language reader: accuracy through eight training passes",
             fontsize=21, weight="bold")
    fig.text(.10, .894, f"One H100 · frozen iteration-2000 player · {count} validation positions from {games} games",
             fontsize=12, color=muted)
    fig.text(.10, .837, f"Selected: pass {selected}", fontsize=15, weight="bold")
    fig.text(.34, .837, f"{facts[selected_i]:.1%} of individual facts correct", fontsize=14, color=teal, weight="bold")
    fig.text(.70, .837, f"{full[selected_i]:.1%} fully correct", fontsize=14, color=blue, weight="bold")
    ax.axvline(selected, color=muted, linestyle=(0, (4, 4)), linewidth=1.2, alpha=.65)
    ax.plot(epochs, facts, color=teal, linewidth=2.8, marker="o", markersize=6,
            label="Individual-fact accuracy")
    ax.plot(epochs, full, color=blue, linewidth=2.8, marker="o", markersize=6,
            label="Full-description accuracy (all four facts correct)")
    ax.scatter([selected, selected], [facts[selected_i], full[selected_i]], s=125,
               facecolor="white", edgecolor=[teal, blue], linewidth=2.4, zorder=5)
    for series, color in ((facts, teal), (full, blue)):
        for i in (0, len(epochs) - 1):
            ax.annotate(f"{series[i]:.1%}", (epochs[i], series[i]), xytext=(0, 13),
                        textcoords="offset points", ha="center", color=color,
                        weight="bold", fontsize=11)
    ax.set(xlim=(-.25, epochs[-1] + .25), ylim=(0, 1),
           xlabel="Additional full passes through the training set (0 = starting reader)",
           ylabel="Validation accuracy")
    ax.set_xticks(epochs)
    ax.set_yticks([i / 5 for i in range(6)])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color=grid)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    time_axis = ax.secondary_xaxis("top")
    time_axis.set_xticks(epochs)
    time_axis.set_xticklabels([f"{minute:.1f}" for minute in minutes])
    time_axis.set_xlabel("Elapsed minutes, including validation", fontsize=11, labelpad=9)
    time_axis.spines["top"].set_visible(False)
    fig.text(.10, .145, "Same held-out validation positions at every point; curves are unsmoothed.",
             fontsize=11, color=muted)
    fig.text(.10, .109, "Four checks: friendly force location, own general location, enemy force visibility/location, army-size balance.",
             fontsize=10, color=muted)
    fig.text(.10, .073, "Selection uses individual-fact accuracy. Full-description accuracy peaks at a different pass.",
             fontsize=10, color=muted)
    fig.text(.10, .037, "Source: Modal grounding run iter2000-grounding-h100-v1 · These are factual-description scores, not game win rates.",
             fontsize=10, color=muted)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"accuracy.{extension}", dpi=180, facecolor="white")
    print(args.output / "accuracy.png")


if __name__ == "__main__":
    main()
