"""Distill a six-tree HAR classifier and evaluate the INT8 CPU graph on held-out readings."""

import argparse
import hashlib
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CHANNELS = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
SUBJECTS = tuple(range(501, 519))
TEST_SUBJECTS = (504, 508, 512, 516)
BATCH_SIZE = 4096


def read_subject(directory, subject):
    path = directory / f"{subject}.csv"
    df = pd.read_csv(path, usecols=CHANNELS + ["label"])
    if len(df) < 5 or df.isna().any().any():
        raise ValueError(f"Missing data or fewer than five rows in {path.name}")
    if not np.isfinite(df[CHANNELS].to_numpy()).all() or not np.issubdtype(df["label"].dtype, np.integer):
        raise ValueError(f"Invalid sensor reading or label in {path.name}")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return df, dict(subject=subject, rows_read=len(df), sha256=digest)


def read_paper_fold(directory):
    """Recreate the first shuffled row fold, including smoothing over CSV boundaries."""
    frames, ids, provenance = [], [], []
    for subject in SUBJECTS:
        df, record = read_subject(directory, subject)
        frames.append(df)
        ids.append(np.full(len(df), subject, dtype=np.int16))
        provenance.append(record)
    merged = pd.concat(frames, ignore_index=True)
    del frames
    x = merged[CHANNELS].rolling(5).mean().iloc[4:].to_numpy(dtype=np.float32)
    y = merged["label"].iloc[4:].to_numpy()
    subject_ids = np.concatenate(ids)[4:]
    fold = KFold(n_splits=10, shuffle=True, random_state=42)
    train, test = next(fold.split(x))
    for record in provenance:
        subject = record["subject"]
        record.update(train_rows=int((subject_ids[train] == subject).sum()),
                      test_rows=int((subject_ids[test] == subject).sum()))
    return {key: (x[idx], y[idx], subject_ids[idx])
            for key, idx in (("train", train), ("test", test))}, provenance


def read_recordings(directory, train_limit, test_limit, seed):
    """Smooth each recording before sampling so windows never span participants."""
    arrays = {"train": [], "test": []}
    provenance = []
    for subject in SUBJECTS:
        df, record = read_subject(directory, subject)
        x = df[CHANNELS].rolling(5).mean().iloc[4:].to_numpy(dtype=np.float32)
        y = df["label"].iloc[4:].to_numpy()
        split = "test" if subject in TEST_SUBJECTS else "train"
        limit = test_limit if split == "test" else train_limit
        if limit and len(y) > limit:
            rng = np.random.default_rng([seed, subject])
            idx = np.sort(rng.choice(len(y), size=limit, replace=False))
            x, y = x[idx], y[idx]
        arrays[split].append((x, y, np.full(len(y), subject, dtype=np.int16)))
        record.update(split=split, rows_used=len(y))
        provenance.append(record)
    return {name: tuple(np.concatenate(v) for v in zip(*parts))
            for name, parts in arrays.items()}, provenance


def export_student(path, scaler, student):
    """Export normalization and the three affine layers, preserving class order."""
    initializers, nodes = [], []

    def tensor(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name

    nodes.append(helper.make_node("Sub", ["features", tensor("mean", scaler.mean_)], ["centered"]))
    nodes.append(helper.make_node("Div", ["centered", tensor("scale", scaler.scale_)], ["standardized"]))
    previous = "standardized"
    for i, (weight, bias) in enumerate(zip(student.coefs_, student.intercepts_)):
        last = i == len(student.coefs_) - 1
        linear = "scores" if last else f"linear_{i}"
        nodes.append(helper.make_node("MatMul", [previous, tensor(f"w{i}", weight)], [f"mm_{i}"]))
        nodes.append(helper.make_node("Add", [f"mm_{i}", tensor(f"b{i}", bias)], [linear]))
        if not last:
            previous = f"hidden_{i}"
            nodes.append(helper.make_node("Relu", [linear], [previous]))
    inputs = [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(CHANNELS)])]
    outputs = [helper.make_tensor_value_info("scores", TensorProto.FLOAT,
                                            [None, student.coefs_[-1].shape[1]])]
    graph = helper.make_graph(nodes, "har70_mlp", inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="har70plus-activity-recognition",
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def infer(path, x):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    scores = np.vstack([session.run(["scores"], {"features": x[pos:pos+BATCH_SIZE]})[0]
                        for pos in range(0, len(x), BATCH_SIZE)])
    if not np.isfinite(scores).all():
        raise ValueError(f"Nonfinite output from {path.name}")
    return scores


def evaluate(y, predicted, classes, ids):
    return dict(accuracy=float(accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, average="macro", labels=classes, zero_division=0)),
                confusion_matrix=confusion_matrix(y, predicted, labels=classes).tolist(),
                accuracy_by_subject={str(i): float(accuracy_score(y[ids == i], predicted[ids == i]))
                                     for i in np.unique(ids)})


def run(args, out):
    print("Reading the 18 recordings", flush=True)
    if args.split == "paper-fold-1":
        data, provenance = read_paper_fold(args.data_dir)
    else:
        data, provenance = read_recordings(args.data_dir, args.train_limit, args.test_limit, args.seed)
    x_train, y_train, _ = data["train"]
    x_test, y_test, ids = data["test"]
    print(f"Fitting RF on {len(x_train):,} rows; {len(x_test):,} held out", flush=True)
    teacher = RandomForestClassifier(n_estimators=6, criterion="gini", random_state=args.seed, n_jobs=2)
    teacher.fit(x_train, y_train)
    classes = teacher.classes_
    if not np.isin(y_test, classes).all():
        raise ValueError("A held-out label is missing from the training set. Increase --train-limit.")
    joblib.dump(teacher, out / "teacher.joblib", compress=0, protocol=5)
    targets = teacher.predict_proba(x_train).astype(np.float32)
    teacher_preds = teacher.predict(x_test)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    student = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                           batch_size=min(1024, len(y_train)), random_state=args.seed,
                           max_iter=args.epochs, tol=0.0, n_iter_no_change=args.epochs,
                           learning_rate_init=0.001, shuffle=True)
    print(f"Fitting the student for {args.epochs} epochs", flush=True)
    with threadpool_limits(limits=1):
        student.fit(train_scaled, targets)
    print("Exporting and evaluating both CPU graphs", flush=True)
    fp32 = out / "student_fp32.onnx"
    int8 = out / "student_int8.onnx"
    export_student(fp32, scaler, student)
    quantize_dynamic(str(fp32), str(int8), op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QInt8, per_channel=True)
    quantized = onnx.load(int8)
    onnx.checker.check_model(quantized)
    integer_layers = sum(node.op_type == "MatMulInteger" for node in quantized.graph.node)
    if integer_layers != len(student.coefs_):
        raise RuntimeError("Quantization did not cover all student matrix multiplications")
    fp_scores = infer(fp32, x_test)
    with threadpool_limits(limits=1):
        native_scores = student.predict(scaler.transform(x_test))
    np.testing.assert_allclose(fp_scores, native_scores, atol=2e-5, rtol=2e-5)
    int_scores = infer(int8, x_test)
    fp_preds, int_preds = (classes[scores.argmax(axis=1)] for scores in (fp_scores, int_scores))
    sample = np.random.default_rng(args.seed).choice(len(x_test), min(4096, len(x_test)), replace=False)
    np.save(out / "benchmark_inputs.npy", x_test[sample])
    np.save(out / "benchmark_labels.npy", y_test[sample])
    (out / "labels.json").write_text(json.dumps(classes.tolist()) + "\n")
    sizes = {f: (out / f).stat().st_size for f in ("teacher.joblib", fp32.name, int8.name)}
    result = {
        "experiment": "Post-publication Random Forest distillation", "split": args.split,
        "runtime": {"python": platform.python_version(), "system": platform.system(),
                    "machine": platform.machine(),
                    "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy",
                                                         "joblib", "onnx", "onnxruntime")}},
        "seed": args.seed, "train_limit_per_subject": args.train_limit,
        "test_limit_per_subject": args.test_limit, "test_subjects": np.unique(ids).tolist(),
        "recordings": provenance, "train_rows": len(y_train), "test_rows": len(y_test),
        "classes": classes.tolist(), "features": CHANNELS, "student_hidden": [64, 32],
        "epochs_completed": student.n_iter_, "inference_batch_rows": BATCH_SIZE,
        "int8_matmul_layers": integer_layers, "providers": ["CPUExecutionProvider"],
        "fp32_export_max_abs_error": float(np.abs(fp_scores - native_scores).max()),
        "teacher": evaluate(y_test, teacher_preds, classes, ids),
        "student_fp32": evaluate(y_test, fp_preds, classes, ids),
        "student_int8": evaluate(y_test, int_preds, classes, ids),
        "int8_teacher_agreement": float((int_preds == teacher_preds).mean()),
        "int8_fp32_agreement": float((int_preds == fp_preds).mean()),
        "model_bytes": sizes, "teacher_to_int8_size_ratio": sizes["teacher.joblib"] / sizes[int8.name],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"INT8 accuracy {result['student_int8']['accuracy']:.4%}; "
          f"teacher {sizes['teacher.joblib']:,} bytes; INT8 {sizes[int8.name]:,} bytes", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, required=True, help="Directory containing 501.csv ... 518.csv")
    cli.add_argument("--output", type=Path, default=Path("artifacts/distillation"))
    cli.add_argument("--split", choices=("participants", "paper-fold-1"), default="participants",
                     help="Hold out four people (default) or use the historical notebook's first row-level fold")
    cli.add_argument("--epochs", type=int, default=12)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--train-limit", type=int, default=0, help="Maximum rows per training participant (0 = all)")
    cli.add_argument("--test-limit", type=int, default=0, help="Maximum rows per test participant (0 = all)")
    args = cli.parse_args()
    if args.epochs < 1 or min(args.seed, args.train_limit, args.test_limit) < 0:
        cli.error("epochs must be positive and seed and row limits must be nonnegative")
    if args.split == "paper-fold-1" and (args.train_limit or args.test_limit):
        cli.error("paper-fold-1 uses all readings. Use the participant split for capped quick checks.")
    if args.output.exists():
        cli.error(f"Output already exists: {args.output}. Choose a new directory.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "result"
        staged.mkdir()
        run(args, staged)
        staged.rename(args.output)


if __name__ == "__main__":
    main()
