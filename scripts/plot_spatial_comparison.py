"""Plot matched validation curves from the two spatial-grounding experiment runs."""
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


def validation_rows(path):
    records = {}
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("stage") == "validation":
            records[row["step"]] = row
    if not records:
        raise ValueError(f"No validation records in {path}")
    return [records[step] for step in sorted(records)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenwise", type=Path, required=True, help="Original adapter's metrics.jsonl")
    parser.add_argument("--spatial", type=Path, required=True, help="Spatial adapter's metrics.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"Original adapter": validation_rows(args.tokenwise),
            "Spatial adapter": validation_rows(args.spatial)}
    positions = {row["metrics"]["real"]["positions"] for rows in runs.values() for row in rows}
    if len(positions) != 1:
        raise ValueError("Validation population sizes differ")
    count = positions.pop()
    args.output.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#183747", "#627c89", "#e4ebee"
    colors = {"Original adapter": "#2875a0", "Spatial adapter": "#098878"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    fig.subplots_adjust(left=.085, right=.965, bottom=.17, top=.79, wspace=.24, hspace=.42)
    fig.text(.085, .945, "Does spatial layout improve the language reader?", fontsize=21, weight="bold")
    fig.text(.085, .901, "Matched B200 runs · 131,072 training snapshots · frozen iteration-2000 player and language model",
             fontsize=11, color=muted)
    fig.text(.085, .86, f"Same {count} validation positions at each checkpoint · unsmoothed accuracy",
             fontsize=11, color=muted)
    measures = [
        ("Fully correct descriptions", lambda m: m["all_four_correct"]),
        ("Individual facts correct", lambda m: m["mean_fact_accuracy"]),
        ("Own general location correct", lambda m: m["per_field_accuracy"]["general_region"]),
        ("Visible enemy location correct", lambda m: m["visible_enemy_location_accuracy"]),
    ]
    max_epoch = max(row["epoch"] for rows in runs.values() for row in rows)
    csv_rows = []
    for axis, (title, measure) in zip(axes.flat, measures):
        for label, rows in runs.items():
            x = [row["epoch"] for row in rows]
            y = [measure(row["metrics"]["real"]) for row in rows]
            axis.plot(x, y, marker="o", linewidth=2.3, markersize=4, label=label, color=colors[label])
            csv_rows.extend((label, title, row["step"], row["epoch"], row["seconds"], value)
                            for row, value in zip(rows, y))
        axis.set(title=title, xlabel="Passes through the training set", ylabel="Accuracy",
                 xlim=(-.08, max(max_epoch, .5) + .08), ylim=(0, 1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color=grid)
        axis.set_axisbelow(True)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(grid)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=10)
    fig.text(.085, .095, "Full accuracy requires all four facts to be correct in the same description. Higher is better.",
             fontsize=10, color=muted)
    fig.text(.085, .058, "Validation selects the checkpoint; the separate final test measures generalization. These are not game win rates.",
             fontsize=10, color=muted)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"accuracy.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    with (args.output / "accuracy.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["adapter", "metric", "step", "pass", "elapsed_seconds", "accuracy"])
        writer.writerows(csv_rows)
    print(args.output / "accuracy.png")


if __name__ == "__main__":
    main()
