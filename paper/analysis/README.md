# Back, thigh and fused accelerometer inputs

This reconstruction asks how much classification accuracy changes when the random forest sees only the lower-back axes, only the thigh axes, or all six. The archive supplies a full-cohort, six-axis [random-forest notebook](../notebooks/classical-models.ipynb). This launcher adds an explicit sensor selection to that experiment and saves a new measurement for each run.

| Configuration | Columns available to the forest |
| --- | --- |
| `back` | `back_x`, `back_y`, `back_z` |
| `thigh` | `thigh_x`, `thigh_y`, `thigh_z` |
| `fused` | All six columns, back followed by thigh |

From the repository root, prepare the [data](../../data/README.md) and install the [paper dependencies](../requirements.txt). Run each configuration separately.

```sh
python paper/analysis/sensor_ablation.py --sensors back
python paper/analysis/sensor_ablation.py --sensors thigh
python paper/analysis/sensor_ablation.py --sensors fused
```

By default each command writes `metrics.json` and the notebook's confusion-matrix figure to `outputs/sensor_ablation/<configuration>/`. To use other locations, pass `--data-dir` and `--output-dir`. An explicit output directory should be unique to that configuration. Per-fold scores and the pooled classification report appear on standard output.

## Interpreting the scores

The launcher executes the original data-loading, smoothing, split and RF cells. It restricts the columns of `X` immediately before the RF cell. All 18 participant files are concatenated in ascending ID order; a trailing five-row mean drops four rows at the start of the combined series. The resulting 2,259,593 row-level predictions are scored out of fold. The shuffled ten-fold split is fixed at 42, and the classifier uses six Gini trees with the notebook's default feature-sampling behavior. The forest itself has no fixed seed, so repeated runs can yield slightly different scores.

Accuracy is the proportion of all pooled out-of-fold predictions that match the observed labels. Macro precision, recall and F1 give every observed activity equal weight. Participants appear in both training and evaluation folds, and adjacent smoothed samples can share raw readings across folds. These scores describe the paper-era row-wise protocol. The [packaging-run results](../results/README.md) identify each new measurement and its source.
