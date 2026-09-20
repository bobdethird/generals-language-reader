"""Plot decision-language accuracy separately from sentence parsing and board facts."""
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


def read_records(path):
    rows = {}
    current = path.parent.name
    for line in path.read_text().splitlines():
        prefix, separator, raw = line.partition('{')
        if not separator:
            continue
        try:
            record = json.loads(separator + raw)
        except json.JSONDecodeError:
            continue
        if record.get("stage") == "start" and "variant" in record:
            current = record["variant"]
        if record.get("stage") != "validation":
            continue
        label = prefix.rstrip(": ") or current
        rows.setdefault(label, {})[record["step"]] = record
    return {label: [steps[key] for key in sorted(steps)] for label, steps in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for path in args.logs:
        runs.update(read_records(path))
    if not runs:
        raise ValueError("No validation records found")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = {"spatial": "Hidden activations", "policy_aux": "+ learned policy head",
              "policy_mask": "+ legal-move mask", "readout": "Original frozen action readout"}
    colors = dict(zip(labels, ("#2475a0", "#8b62b8", "#c2782c", "#0b8d7e")))
    ink, muted, grid = "#193745", "#627b87", "#e3eaee"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white")
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.18, hspace=.5, wspace=.24)
    fig.text(.09, .94, "Can the language reader preserve the player's decision?", fontsize=20, weight="bold")
    fig.text(.09, .895, "Frozen iteration-2000 player · automatic supervision · 131,072 training snapshots", color=muted)
    fig.text(.09, .85, "Same 512 held-out validation positions · curves select checkpoints; a separate test follows", color=muted)
    measures = [("Exact move, including source and troop split", lambda m: m["exact_action_accuracy"]),
                ("Complete description correct", lambda m: m["fully_correct_description"]),
                ("Source square correct on movement positions", lambda m: m["move_field_accuracy"]["source"]),
                ("Direction correct on movement positions", lambda m: m["move_field_accuracy"]["direction"])]
    csv_rows = []
    for axis, (title, measure) in zip(axes.flat, measures):
        for key, rows in sorted(runs.items()):
            variant = next((v for v in labels if key.endswith('-'+v) or key == v), key)
            label = labels.get(variant, key)
            color = colors.get(variant, "#627b87")
            axis.plot([r["epoch"] for r in rows], [measure(r["metrics"]["real"]) for r in rows],
                      marker="o", linewidth=2, markersize=4, color=color, label=label)
            csv_rows.extend((label, title, r["step"], r["epoch"], measure(r["metrics"]["real"])) for r in rows)
        axis.set(title=title, xlabel="Passes through training data", ylabel="Accuracy", ylim=(0, 1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color=grid)
        axis.set_axisbelow(True)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(grid)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(.09, .10, "Exact move accuracy matches the original raw-logit greedy rule. Random sampled moves are a different target.",
             color=muted, fontsize=10)
    fig.text(.09, .06, "A correct description reports the move and observable context; it does not establish a strategic motive.",
             color=muted, fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"decision-accuracy.{extension}", dpi=180)
    plt.close(fig)
    with (args.output/"decision-accuracy.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["reader", "metric", "step", "pass", "accuracy"])
        writer.writerows(csv_rows)
    print(args.output/"decision-accuracy.png")


if __name__ == "__main__":
    main()
