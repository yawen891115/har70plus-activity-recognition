"""Predict activity IDs with the exported CPU model and print CSV to stdout."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from demo_inputs import demo_features

HERE = Path(__file__).resolve().parent
CHANNELS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
BATCH_SIZE = 4096


def prepare_input(path):
    if path.suffix.lower() == ".csv":
        import pandas as pd

        frame = pd.read_csv(path, usecols=CHANNELS)
        values = frame[CHANNELS].rolling(5).mean().iloc[4:].to_numpy(dtype=np.float32)
        return values, 4
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32), 0
    raise ValueError("Input must be a recording (.csv) or six-channel smoothed features (.npy)")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", type=Path, default=HERE / "reference")
    cli.add_argument("--input", type=Path, help="One raw recording (CSV) or smoothed features (NPY)")
    cli.add_argument("--limit", type=int, help="Print at most this many predictions")
    args = cli.parse_args()
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    if args.input is None:
        values, offset = demo_features(), 0
    else:
        values, offset = prepare_input(args.input)
    if values.ndim != 2 or values.shape[1] != len(CHANNELS) or not len(values) or not np.isfinite(values).all():
        cli.error("Input must contain one or more finite six-channel feature rows after smoothing")
    labels = np.array(json.loads((args.model_dir / "labels.json").read_text()))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.model_dir / "student_int8.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["input_row_zero_based", "activity_label"])
    count = len(values) if args.limit is None else min(len(values), args.limit)
    for start in range(0, count, BATCH_SIZE):
        scores = session.run(["scores"], {"features": values[start:start+BATCH_SIZE]})[0]
        for i, label in enumerate(labels[scores.argmax(axis=1)][:count-start]):
            writer.writerow([offset + start + i, int(label)])


if __name__ == "__main__":
    main()
