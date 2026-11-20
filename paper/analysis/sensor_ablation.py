"""Reconstruct one sensor configuration using the archived full-cohort RF cells."""

import argparse
import json
import platform
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER / "notebooks"))
from run_model import run_cells


SENSORS = {
    "back": ["back_x", "back_y", "back_z"],
    "thigh": ["thigh_x", "thigh_y", "thigh_z"],
}
SENSORS["fused"] = SENSORS["back"] + SENSORS["thigh"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensors", choices=SENSORS, required=True)
    parser.add_argument("--data-dir", type=Path, default=PAPER.parent / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, help="directory for one configuration's figure and metrics")
    args = parser.parse_args()
    output_dir = args.output_dir or PAPER.parent / "outputs" / "sensor_ablation" / args.sensors
    features = SENSORS[args.sensors]
    print(f"RF reconstruction with {args.sensors} sensors: {', '.join(features)}", flush=True)
    try:
        state = run_cells("rf", args.data_dir, output_dir, features=features)
    except ValueError as error:
        parser.error(str(error))

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    actual, predicted = state["all_y_true"], state["all_y_pred"]
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="macro")
    report = {
        "protocol": "archival RF configuration, fresh sensor-ablation run",
        "sensors": args.sensors,
        "features": features,
        "participant_ids": list(range(501, 519)),
        "evaluated_rows": len(actual),
        "kfold": {"n_splits": 10, "shuffle": True, "random_state": 42, "unit": "row"},
        "random_forest": {"n_estimators": 6, "criterion": "gini", "random_state": None},
        "pooled_out_of_fold": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "labels": [int(label) for label in state["labels"]],
            "confusion_matrix": state["cm"].tolist(),
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    metrics = output_dir / "metrics.json"
    metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {metrics}", flush=True)


if __name__ == "__main__":
    main()
