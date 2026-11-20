# Research notebooks

The notebooks retain the author's source cells and execution counts. The public copies of the descriptive-statistics, sensor-denoising and exploratory notebooks keep selected aggregate outputs. Re-running their cells locally produces the participant-level tables and plots. The filenames below identify their contents. Cell numbers count all cells from the top, including Markdown.

| Notebook | Original filename | What to inspect |
| --- | --- | --- |
| [Descriptive statistics](descriptive-statistics.ipynb) | 描述性统计.ipynb | Cell 1 merges all 18 files and prints 2,259,597 input observations. Cell 2 counts the seven labels after the five-sample rolling mean. Cells 3–4 plot each axis by activity for the back and thigh. |
| [Sensor denoising](sensor-denoising.ipynb) | 降噪（和向量版）.ipynb | Cells 1–5 show raw and smoothed acceleration magnitudes for participant 504 (back) and 512 (thigh). Cell 3's axis label still says 504. Cell 6 computes correlations and variance inflation factors on participants 501 and 502. |
| [Classical models](classical-models.ipynb) | 前四个模型4.5.ipynb | Full-cohort setup, then random forest, logistic regression, QDA, KNN and Gaussian naive Bayes. The notebook contains training and evaluation code without saved outputs. |
| [Neural models](neural-models.ipynb) | NN45.ipynb | Full-cohort Keras MLP, CNN and LSTM experiments. The notebook contains training and evaluation code without saved outputs. |
| [Result figures](result-figures.ipynb) | 图像代码.ipynb | Cells 1 and 3 plot manually entered comparison values. Cell 4 plots ten entered losses per neural model. Cells 5–6 label their confusion-matrix inputs as placeholders or assumptions. |
| [Exploratory workbook](exploratory-workbook.ipynb) | har70垃圾中找黄金.ipynb | Cell 20 records 82.79% mean shuffled ten-fold accuracy for logistic regression on the raw rows of participants 501 and 502. Later cells depend on earlier kernel state. The workbook also contains an unrelated smoking-classification outline and CNN trial, plus interrupted fits. |

The accompanying [Chinese metric notes](../notes/metric-formulas-zh.docx) came from `latex版本的某些公式-翻译用.docx`. These are drafting notes. In the model code, `accuracy_score` computes the fraction of correct predictions. The draft's accuracy expression sums true negatives across one-vs-rest problems and can give a different number in multiclass classification.

## Run a full-cohort model

From the repository root, use the [paper environment](../README.md) and the [18 input CSV files](../../data/README.md).

```bash
python paper/notebooks/run_model.py rf
```

Progress, fold metrics and the pooled classification report appear on standard output. The runner saves the figure in `outputs/paper/`. Both `--data-dir` and `--output-dir` accept other locations. Each invocation starts a fresh kernel-equivalent Python namespace, executes the specified original cells, and redirects their file paths in memory. It sets a noninteractive plotting backend by default.

| Command argument | Notebook cells | Evaluation in those cells |
| --- | --- | --- |
| `rf` | Classical, 2 → 4 → 8 | Six Gini trees. Shuffled ten-fold cross-validation on rows with split seed 42. The forest has no fixed random seed, so reruns can differ. |
| `logistic` | Classical, 2 → 4 → 10 | Five unshuffled row folds inherited from cell 8. Multinomial logistic regression, SAG solver, maximum 1,000 iterations. |
| `qda` | Classical, 2 → 4 → 12 | Five unshuffled row folds. Quadratic discriminant analysis with library defaults. |
| `nb` | Classical, 2 → 4 → 16 | Five unshuffled row folds. Gaussian naive Bayes with library defaults. Writes `NB.png` so the original output name (`QDA.png`) cannot overwrite the QDA figure. |
| `mlp` | Neural, 1 → 3 | Row-level 70/30 split with seed 42. Two 64-unit dense layers, ten epochs, batch size 32. Validation uses 20% of the training split. |
| `cnn` | Neural, 1 → 5 | Row-level 70/30 split with seed 15. Each row is reshaped to 2 × 3 × 1. Ten epochs, batch size 64. The test split also supplies validation data during training. |
| `lstm` | Neural, 1 → 7 | Row-level 70/30 split with seed 42. One timestep per row, LSTM layers with 128 and 64 units and dropout of 0.5. Up to ten epochs with early stopping. |

All seven paths apply the five-row moving average to the concatenated sensor channels before splitting. Each participant's rows can appear in both training and evaluation. A rolling window can span a participant boundary, and adjacent filtered observations can share source rows across splits. The resulting metrics describe row-level classification in this cohort.

The classical workbook initially sets ten shuffled folds in cell 4. Cell 8 resets the splitter to five unshuffled folds **after** collecting the forest's predictions. The logistic, QDA and naive Bayes commands restore that five-fold splitter before executing the selected training cell. This reproduces the fold assignments obtained by running the notebook from top to bottom, without first fitting the preceding classifiers. Historical cell execution order is unknown because this workbook has no saved outputs. The standalone scripts have their own protocols. The neural MLP and CNN cells each convert labels to one-hot encoding in place. The launcher executes each model with freshly loaded labels, preserving its individual experiment.

The three neural model commands completed training, evaluation and figure export in a CPU smoke test with Python 3.11.18, TensorFlow 2.15.1 and scikit-learn 1.5.2. That test used 2,520 synthetic observations in 18 CSV files to verify inputs, model shapes and outputs. On the same fixture, the logistic, QDA and naive Bayes commands matched all 2,516 evaluated row predictions and all three rendered matrices from a sequential run of the original classical workbook with the same random states.

## Values used to draw the comparison figure

These values come directly from cell 1 of [Result figures](result-figures.ipynb). They are plotting inputs transcribed in the original notebook. Re-running the figure only redraws the bars. For new measurements, run the training cells and retain their standard output.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7670 | 0.4255 | 0.4256 | 0.4078 |
| Gaussian naive Bayes | 0.8290 | 0.5525 | 0.5159 | 0.4937 |
| Random forest | 0.9428 | 0.8462 | 0.6354 | 0.6776 |
| Quadratic discriminant analysis | 0.8430 | 0.5582 | 0.5342 | 0.5181 |
| MLP | 0.9294 | 0.7870 | 0.5751 | 0.5852 |
| CNN | 0.9278 | 0.7931 | 0.5743 | 0.5829 |
| LSTM | 0.9217 | 0.6745 | 0.5520 | 0.5413 |
