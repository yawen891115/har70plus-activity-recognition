"""Regression checks for the packaged ONNX graphs, label mapping and CSV input boundary."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from demo_inputs import demo_features
from predict import CHANNELS, prepare_input
from run import SUBJECTS, read_paper_fold
from sklearn.model_selection import KFold


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"


class ReleaseTests(unittest.TestCase):
    def test_rolling_input_column_order_and_alignment(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "subject.csv"
            channel_values = np.arange(42, dtype=np.float32).reshape(7, 6)
            frame = pd.DataFrame(channel_values, columns=CHANNELS)
            frame[list(reversed(CHANNELS))].to_csv(path, index=False)
            features, offset = prepare_input(path)
        self.assertEqual(offset, 4)
        expected = np.stack([channel_values[i:i + 5].mean(axis=0) for i in range(3)])
        np.testing.assert_array_equal(features, expected)

    def test_first_fold_matches_archived_window_and_split(self):
        frames = []
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            for subject in SUBJECTS:
                values = np.arange(66, dtype=float).reshape(11, 6) + subject
                frame = pd.DataFrame(values, columns=CHANNELS)
                frame["label"] = np.arange(len(frame)) + 1000 * subject
                frame.to_csv(directory / f"{subject}.csv", index=False)
                frames.append(frame)
            data, _ = read_paper_fold(directory)
        merged = pd.concat(frames, ignore_index=True)
        for channel in CHANNELS:
            merged[channel] = merged[channel].rolling(5).mean()
        merged = merged.iloc[4:].reset_index(drop=True)
        train, test = next(KFold(n_splits=10, shuffle=True, random_state=42).split(merged))
        for name, indices in (("train", train), ("test", test)):
            x, y, ids = data[name]
            expected = merged.iloc[indices]
            np.testing.assert_array_equal(x, expected[CHANNELS].to_numpy(dtype=np.float32))
            np.testing.assert_array_equal(y, expected["label"].to_numpy())
            np.testing.assert_array_equal(ids, y // 1000)

    def test_released_sizes_and_execution(self):
        for directory in (REFERENCE, REFERENCE / "paper-fold-1"):
            with self.subTest(directory=directory.name):
                self.check_release(directory)

    def check_release(self, directory):
        info = json.loads((directory / "metrics.json").read_text())
        label_ids = np.array(json.loads((directory / "labels.json").read_text()))
        self.assertEqual(info["classes"], label_ids.tolist())
        sample = demo_features()
        for name in ("teacher", "student_fp32", "student_int8"):
            metrics = info[name]
            matrix = np.asarray(metrics["confusion_matrix"])
            self.assertEqual(matrix.shape, (len(label_ids), len(label_ids)))
            self.assertEqual(matrix.sum(), info["test_rows"])
            self.assertAlmostEqual(matrix.trace() / matrix.sum(), metrics["accuracy"])
            denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
            f1 = np.divide(2 * matrix.diagonal(), denominator,
                           out=np.zeros(len(label_ids)), where=denominator != 0)
            self.assertAlmostEqual(f1.mean(), metrics["macro_f1"])
        for name in ("student_fp32", "student_int8"):
            graph = directory / f"{name}.onnx"
            self.assertEqual(graph.stat().st_size, info["model_bytes"][graph.name])
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(graph), sess_options=opts, providers=["CPUExecutionProvider"])
            scores = sess.run(["scores"], {"features": sample})[0]
            self.assertEqual(scores.shape, (len(sample), len(label_ids)))
            self.assertTrue(np.isfinite(scores).all())

    def test_prediction_cli(self):
        for directory in (REFERENCE, REFERENCE / "paper-fold-1"):
            model_args = ["--model-dir", str(directory)]
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(directory / "student_int8.onnx"), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            scores = sess.run(["scores"], {"features": demo_features()})[0]
            labels = np.array(json.loads((directory / "labels.json").read_text()))
            expected = labels[scores[:8].argmax(axis=1)].tolist()
            with self.subTest(model_args=model_args):
                command = [sys.executable, str(HERE / "predict.py"), "--limit", "8", *model_args]
                rows = subprocess.check_output(command, text=True).splitlines()
                self.assertEqual(rows[0], "input_row_zero_based,activity_label")
                self.assertEqual([int(row.split(",")[0]) for row in rows[1:]], list(range(8)))
                self.assertEqual([int(row.split(",")[1]) for row in rows[1:]], expected)


if __name__ == "__main__":
    unittest.main()
