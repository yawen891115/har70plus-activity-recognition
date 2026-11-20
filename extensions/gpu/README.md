# ONNX Runtime graph optimizations on CUDA

This optional follow-up measures the effect of ONNX Runtime graph optimizations on the floating-point MLP exported by [the distillation experiment](../distillation/). Both configurations use the same CUDA device, FP32 graph with TensorFloat-32 disabled, and sequence of input batches. The paper's classifiers and the CPU INT8 deployment have separate entry points.

## Run

The benchmark loads the bundled participant-holdout FP32 student by default. Generate 4,096 synthetic rows locally with the command below, or pass your own feature array to `--inputs`. Use `--model` to select a freshly trained or row-fold checkpoint. Both accept the six smoothed float32 features in the export order. Use a separate Python 3.11+ environment with a supported NVIDIA GPU, CUDA and cuDNN versions matching the installed ONNX Runtime GPU wheel. See the [ONNX Runtime CUDA compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

From the repository root:

```bash
python -m pip install -r extensions/gpu/requirements.txt
python extensions/distillation/demo_inputs.py artifacts/synthetic-inputs.npy
python extensions/gpu/benchmark.py --inputs artifacts/synthetic-inputs.npy \
  --batch-size 64 --warmup 20 --runs 200 \
  --output artifacts/gpu/batch64.json --traces artifacts/gpu/traces
```

For single-reading inference, repeat with `--batch-size 1 --output artifacts/gpu/batch1.json`. The default output is `artifacts/gpu/benchmark.json`. The timing loop cycles over full batches and alternates which configuration runs first. The output comparison covers **all** provided rows, including the last partial batch.

## Read the results

| Field | Interpretation |
| --- | --- |
| `disabled`, `enabled` | Graph optimizations disabled or fully enabled. Each uses an untimed warm-up and a separate untimed placement trace to verify the MLP's matrix multiplication ran on CUDA. |
| `p50_batch_ms`, `p95_batch_ms` | Wall-clock latency per synchronous inference call, including transfers from host memory to the GPU and back. Session creation and data loading are outside the timed region. |
| `rows_per_second` | Total timed rows divided by total measured call time for that configuration. |
| `p50_speedup` | Baseline p50 divided by optimized p50. A value below 1 indicates a slowdown at that batch size. |
| `output_comparison` | Top-1 agreement between configurations and maximum absolute class-score difference on the supplied sample. |
| `observed_node_ops_by_provider` | Operator types observed on each execution provider in the placement checks. |

The release was validated on CPU. GPU timings are pending a run on NVIDIA hardware. CPU quantization, serialized model size, and ground-truth classification accuracy are evaluated in the distillation experiment.
