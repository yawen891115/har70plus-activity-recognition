"""Plot per-activity recall from saved full-cohort confusion matrices."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULTS / "metrics.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "sensor-recall.png")
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    names = data["activity_names"]
    columns = {"back": "Back only", "thigh": "Thigh only", "fused": "Back + thigh"}
    recalls = []
    for location in columns:
        cm = np.array(data["results"][location]["confusion_matrix"])
        recalls.append(cm.diagonal() / cm.sum(axis=1))
    recall = np.array(recalls).T

    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    ax.imshow(recall, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns.values(), fontsize=11)
    ax.set_yticks(range(len(names)), names, fontsize=10)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.xaxis.tick_top()
    ax.set_title("Recall by activity and sensor location", fontsize=16, pad=16, loc="left")
    for row, values in enumerate(recall):
        for col, value in enumerate(values):
            color = "white" if value > 0.65 else "#153451"
            ax.text(col, row, f"{value:.1%}", va="center", ha="center", color=color, fontsize=10)
    ax.set_yticks(np.arange(0.5, len(names) - 0.5), minor=True)
    ax.set_xticks(np.arange(0.5, len(columns) - 0.5), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
