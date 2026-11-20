# Random Forest distillation on CPU

This follow-up fits a new six-tree Random Forest and uses its class probabilities to train a compact MLP. The INT8 student runs on ONNX Runtime's CPU execution provider. The original RF configuration appears in [the classical-model notebook](../../paper/notebooks/classical-models.ipynb).

## Measured results

The full run trained on 1,720,701 smoothed rows and evaluated 538,824 rows from four unseen participants. The model-size column measures actual serialized bytes.

| Model | Accuracy | Macro F1 | Model size |
| --- | ---: | ---: | ---: |
| Random Forest teacher (joblib, uncompressed) | 77.34% | 0.4772 | 97,808,705 B |
| FP32 MLP (ONNX) | 79.10% | 0.4725 | 11,605 B |
| INT8 MLP (ONNX Runtime CPU) | 79.45% | 0.4747 | 6,101 B |

The INT8 model agreed with the teacher on 91.69% of evaluation rows. Compare the two students to isolate the export and quantization effect. Comparing the teacher with either student also changes the model architecture and serialization format. The [full run record](reference/metrics.json) includes confusion matrices and accuracy per participant.

| Evaluation split | Training rows | Evaluation rows | RF accuracy | FP32 accuracy | INT8 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four unseen participants | 1,720,701 | 538,824 | 77.34% | 79.10% | 79.45% |
| Historical row split, fold 1 only | 2,033,633 | 225,960 | 94.24% | 92.74% | 92.61% |

The row-level result is a single held-out fold. It measures discrimination among readings from the same pool of participants. The notebook's original RF result aggregates predictions from ten folds.

## Predict with the released student

From the repository root, in a Python 3.12 environment:

```bash
python -m pip install numpy==2.3.5 onnxruntime==1.22.1
python extensions/distillation/predict.py --limit 8
```

The default command runs the participant-holdout model on deterministic synthetic features. The demonstration has no ground-truth labels. To run the 92.61% row-fold model on the same inputs, select its model directory.

```bash
python extensions/distillation/predict.py --model-dir extensions/distillation/reference/paper-fold-1 --limit 8
```

Both commands print activity IDs. To predict from one raw participant recording using the default model, install pandas and run:

```bash
python -m pip install pandas==2.2.3
python extensions/distillation/predict.py --input data/raw/504.csv --limit 8
```

Each prediction corresponds to the last reading of a trailing five-row window. Source rows are numbered from zero, so the first prediction for a raw CSV refers to row 4. With NPY input the script expects already-smoothed values in the order shown below. Class IDs follow the [dataset reference](../../data/README.md).

## Train and evaluate

```bash
python -m pip install -r extensions/distillation/requirements.txt
python extensions/distillation/run.py --data-dir data/raw --output artifacts/distillation
```

For a matched row-level comparison, use fold 1 of the archived notebook's shuffled ten-fold split:

```bash
python extensions/distillation/run.py --data-dir data/raw --split paper-fold-1 --output artifacts/distillation-paper-fold-1
```

This option uses all 2,259,597 readings in ascending participant order, applies the original five-reading moving mean across the concatenated series and drops four rows at its start. It then evaluates only the first fold of `KFold(10, shuffle=True, random_state=42)`. Participant IDs can appear on both sides, including nearby rows with overlapping smoothing windows. The teacher and student are trained afresh with the same hyperparameters as the participant-holdout extension. The top-level files in `reference/` describe the participant holdout. The [first-fold artifacts](reference/paper-fold-1/) include the separately trained student, its label map and evaluation record. See the [first-fold run record](reference/paper-fold-1/metrics.json) for exact metrics.

For a quick check of the entire training, export and scoring path:

```bash
python extensions/distillation/run.py --data-dir data/raw --train-limit 1000 --test-limit 500 --epochs 2 --output artifacts/distillation-smoke
```

Each command writes to a fresh directory. The full default command reproduces the participant-holdout protocol below. The quick check uses a seeded random sample per person and two training epochs.

| Component | Protocol |
| --- | --- |
| Features | Five-reading trailing mean of back x/y/z followed by thigh x/y/z. Each recording starts a new window sequence. The first four rows per person are omitted. |
| Split | Train on 14 participants. Hold out 504, 508, 512 and 516 in their entirety. The IDs are fixed in the script. This extension estimates transfer to unseen participants using one holdout. The paper used row-level cross-validation. |
| Teacher | Six Gini decision trees using scikit-learn's Random Forest defaults, with seed 42. |
| Targets | Teacher class probabilities on training rows, ordered by the teacher's numeric class IDs. Each row's target is the distribution across all seven classes. Held-out labels are used only for evaluation. |
| Student | Standardize using training rows. Fit a 6–64–32–7 MLP with ReLU hidden layers and a linear output by minimizing squared error to the soft targets. Adam, 0.001 learning rate, shuffled minibatches of 1,024, twelve fixed epochs. |
| Export | The FP32 graph accepts smoothed features and includes standardization. The INT8 graph dynamically quantizes activations and uses per-channel signed INT8 weights for all three matrix multiplications. Other operations stay in float. Both graphs return class scores, with the label selected by argmax. |
| Scoring | Evaluate the RF, native MLP and both exported graphs on the same held-out input. Check native vs FP32 scores numerically and confirm three quantized matrix multiplications. Report accuracy, macro F1, per-subject accuracy, confusion matrices and model agreement. Dynamic quantization scales depend on the input batch, so inference uses batches of 4,096 rows. |
| Size comparison | Actual byte length of the uncompressed joblib teacher (protocol 5) divided by the standalone INT8 ONNX graph size. This excludes the runtime, the training environment and the small external label-map file. |

[Reference run and model files](reference/) include `metrics.json`, the exported FP32 and INT8 models and the ordered label map. The full training command also writes a teacher and sampled benchmark inputs and labels locally. The six columns in `benchmark_inputs.npy` follow the order `back_x, back_y, back_z, thigh_x, thigh_y, thigh_z`. Use these inputs for CPU or GPU inference. The saved metrics refer to the complete holdout under the original batching of that run.

The public demonstration uses uniform pseudorandom inputs generated by [demo_inputs.py](demo_inputs.py). The reported evaluation metrics use the [HAR70+ recordings](../../data/README.md).

To verify the released files and input handling, run `python extensions/distillation/test_release.py` from the repository root after installing the full distillation requirements.
