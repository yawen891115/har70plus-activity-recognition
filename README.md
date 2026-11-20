# Human activity recognition with back and thigh accelerometers

Code accompanying **Tri-axial Accelerometer Data-driven Senior Activity Recognition: Multi-class Machine Learning Approaches** by Yawen Zheng and Junping Wei (IEEE ICID 2024). The study compares classifiers for seven activities using 2,259,597 labeled readings from 18 adults aged 70–95. Each reading pairs three acceleration axes from the lower back with three from the right thigh at 50 Hz. [Paper](https://doi.org/10.1109/ICID64166.2024.11024862) · [HAR70+ dataset](https://doi.org/10.24432/C5CW3D)

The full-cohort rerun measured **94.26% accuracy** using both sensors, compared with **85.17%** for back only and **87.68%** for thigh only. [Per-activity results and original score records](paper/results/README.md)

## Experiments

| Track | What you can run | Start here |
| --- | --- | --- |
| **1 · Original study** | Compare classical and neural models. The full-cohort random forest uses six trees, a five-sample moving average and shuffled ten-fold cross-validation on readings. The MLP, CNN and LSTM each use a 70/30 holdout. | [Full-cohort notebooks and settings](paper/notebooks/README.md) · [Original scripts](paper/README.md) |
| **1 · Sensor comparison** | Repeat the full-cohort RF experiment with back axes, thigh axes or both. A separate launcher selects the input columns and executes the original model cells. | [Ablation procedure](paper/analysis/README.md) · [Study results](paper/results/README.md) |
| **2 · CPU distillation** | Train a small MLP on a forest's class probabilities, quantize the student to INT8, and run inference with ONNX Runtime. A separate evaluation holds out four entire participants. | [Training and inference](extensions/distillation/README.md) · [Weights and evaluation](extensions/distillation/reference/) |
| **2 · GPU profiling** | Compare the FP32 student's CUDA inference with graph optimizations disabled and enabled. Measure p50/p95 latency, throughput and prediction agreement on identical inputs. | [CUDA benchmark](extensions/gpu/README.md) |

## Reproduce the study's random forest

In a fresh Python 3.11 environment, run these commands from the repository root.

```bash
python -m pip install -r paper/requirements.txt
python data/prepare.py
python paper/notebooks/run_model.py rf
```

The run covers IDs 501–518 and prints fold metrics and a pooled classification report. The same launcher supports `logistic`, `nb`, `qda`, `mlp`, `cnn` and `lstm`; the [notebook guide](paper/notebooks/README.md) lists each model’s evaluation split. The standalone `har70RF.py` is a separate two-participant experiment. The [data guide](data/README.md) includes offline preparation from the original archive. All 18 participant files are verified against their original SHA-256 hashes.

## Try the distilled model

In a fresh Python 3.12 environment, run the bundled INT8 graph on eight synthetic feature vectors generated with a fixed random seed. The recorded accuracy comes from separate evaluations on HAR70+. Each released INT8 graph occupies **6,101 bytes**. The participant-holdout model attained **79.45%** on four unseen participants. A second model attained **92.61%** on the first of the original shuffled row-level folds.

```bash
python -m pip install numpy==2.3.5 onnxruntime==1.22.1
python extensions/distillation/predict.py --limit 8
```

To classify a raw participant CSV or retrain the teacher and student, follow the [CPU instructions](extensions/distillation/README.md). For the optional GPU experiment, the [benchmark instructions](extensions/gpu/README.md) give the hardware requirements and timing protocol.

## Archive and attribution

The [source inventory](paper/source-manifest.tsv) maps all 19 Python scripts, six notebooks and the original formula notes to their filenames in the research archive. All 19 scripts and all notebook source cells preserve the original code and Chinese comments. Three public notebooks omit saved outputs containing individual readings or traces. The inventory records the original and public file hashes. The launchers resolve the research machine's local paths at runtime. [Per-file study guide](paper/README.md) · [Notebook guide](paper/notebooks/README.md)

HAR70+ was provided by Aleksej Logacjov and Astrid Ustad under CC BY 4.0. The dataset and study have separate citations. [Data credits](data/README.md) · [Study citation](CITATION.cff)
