"""Benchmark ONNX Runtime graph optimizations on the distilled FP32 HAR student."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np


CUDA = "CUDAExecutionProvider"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = Path(__file__).resolve().parents[1] / "distillation" / "reference"
    parser.add_argument("--model", type=Path, default=source / "student_fp32.onnx")
    parser.add_argument("--inputs", type=Path, required=True, help="Local float32 [rows, 6] .npy features")
    parser.add_argument("--output", type=Path, default=Path("artifacts/gpu/benchmark.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200, help="Timed calls per configuration.")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device ordinal.")
    parser.add_argument("--traces", type=Path, help="Keep ORT placement traces in this directory.")
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.runs) < 1 or args.device < 0:
        parser.error("Batch size, warm-up, and runs must be positive; device must be nonnegative.")
    return args


def placement(trace):
    nodes = defaultdict(set)
    for event in json.loads(trace.read_text()):
        details = event.get("args", {})
        if event.get("cat") == "Node" and details.get("provider") and details.get("op_name"):
            nodes[details["provider"]].add(details["op_name"])
    if not any("gemm" in op.lower() or "matmul" in op.lower() for op in nodes[CUDA]):
        raise RuntimeError("Trace has no CUDA MLP matrix multiplication. Check the graph and CUDA setup.")
    return {provider: sorted(ops) for provider, ops in nodes.items()}


def make_session(ort, args, level, trace_prefix=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    if trace_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(trace_prefix)
    session = ort.InferenceSession(
        str(args.model), sess_options=options,
        providers=[(CUDA, {"device_id": args.device, "use_tf32": "0"}), "CPUExecutionProvider"],
    )
    session.disable_fallback()
    if CUDA not in session.get_providers():
        raise RuntimeError("ONNX Runtime failed to initialize its CUDA provider.")
    return session


def scores(session, batch):
    values, = session.run(None, {"features": batch})
    return values


def score_drift(sessions, inputs, size):
    matches = 0
    largest = 0.0
    for start in range(0, len(inputs), size):
        part = inputs[start:start + size]
        a, b = [scores(session, part) for session in sessions.values()]
        if (a.ndim != 2 or a.shape[0] != len(part) or a.shape[1] < 2 or a.shape != b.shape
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("Expected finite [batch, classes] scores in both configurations.")
        largest = max(largest, float(np.max(np.abs(a - b))))
        matches += int(np.count_nonzero(a.argmax(axis=1) == b.argmax(axis=1)))
    return {"rows": len(inputs), "argmax_matches": matches, "argmax_agreement": matches / len(inputs),
            "max_abs_score_delta": largest}


def gpu_inventory():
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            result = subprocess.run(
                [exe, "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=8,
            )
            return result.stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def fingerprint(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def benchmark(ort, args, trace_dir):
    if CUDA not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable. Install the GPU runtime on a CUDA host.")
    inputs = np.load(args.inputs, allow_pickle=False)
    if (inputs.ndim != 2 or inputs.shape[1] != 6 or inputs.dtype != np.float32
            or not np.isfinite(inputs).all() or len(inputs) < args.batch_size):
        raise ValueError("Expected a finite float32 [rows, 6] input array with at least one full batch.")
    inputs = np.ascontiguousarray(inputs)
    levels = {"disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
              "enabled": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
    evidence, sessions = {}, {}
    for name, level in levels.items():
        traced = make_session(ort, args, level, trace_dir / name)
        scores(traced, inputs[:args.batch_size])
        trace = Path(traced.end_profiling())
        evidence[name] = placement(trace)
        del traced
        sessions[name] = make_session(ort, args, level)

    times = {name: [] for name in sessions}
    n_batches = len(inputs) // args.batch_size
    names = list(sessions)
    for i in range(args.warmup + args.runs):
        start = (i % n_batches) * args.batch_size
        batch = inputs[start:start + args.batch_size]
        for name in (names if i % 2 == 0 else reversed(names)):
            begin = time.perf_counter_ns()
            # session.run returns host NumPy output. Timings include completed H2D, compute, and D2H.
            scores(sessions[name], batch)
            elapsed_ns = time.perf_counter_ns() - begin
            if i >= args.warmup:
                times[name].append(elapsed_ns / 1e6)

    timings = {}
    for name, latency in times.items():
        timings[name] = {
            "p50_batch_ms": float(np.percentile(latency, 50)),
            "p95_batch_ms": float(np.percentile(latency, 95)),
            "rows_per_second": args.batch_size * args.runs * 1000 / sum(latency),
        }
    return {
        "onnxruntime_version": ort.__version__, "numpy_version": np.__version__,
        "platform": platform.platform(), "cuda_device_ordinal": args.device, "gpu_inventory": gpu_inventory(),
        "model_sha256": fingerprint(args.model), "inputs_sha256": fingerprint(args.inputs),
        "precision": "FP32 graph, CUDA TF32 disabled", "graph_levels": ["ORT_DISABLE_ALL", "ORT_ENABLE_ALL"],
        "batch_size": args.batch_size, "warmup_per_config": args.warmup, "timed_calls_per_config": args.runs,
        "timed_rows_per_config": args.runs * args.batch_size,
        "scope": "synchronous host inputs to host class scores; single-process sequential calls",
        "observed_node_ops_by_provider": evidence, "timings": timings,
        "p50_speedup": timings["disabled"]["p50_batch_ms"] / timings["enabled"]["p50_batch_ms"],
        "output_comparison": score_drift(sessions, inputs, args.batch_size),
    }


def main():
    args = arguments()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Install extensions/gpu/requirements.txt in a CUDA environment.") from exc
    if args.traces:
        args.traces.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            report = benchmark(ort, args, args.traces or Path(temporary))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    Path(out.name).replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
