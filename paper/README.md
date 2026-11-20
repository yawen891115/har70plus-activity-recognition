# Paper-era experiments

The Python files in [scripts](scripts/) are the author's original files. The archive also contains [research notebooks](notebooks/) with different participant ranges and, in some cases, different inputs. For the **18-participant random-forest experiment**, start with the RF section of [the classical-models notebook](notebooks/classical-models.ipynb). The standalone `har70RF.py` reads participants **501 and 502**.

## Run a standalone script

From the repository root, prepare the CSVs using the [data instructions](../data/README.md). The commands below assume Python 3.10 or 3.11 and a fresh virtual environment.

```bash
python -m pip install -r paper/requirements.txt
python paper/run.py har70NB.py
python paper/run.py har70RF.py
python paper/run.py "stat descri.py" --output-dir outputs/paper
```

Install `paper/requirements-neural.txt` for the Keras scripts. These two requirements files describe compatible historical APIs. The archive contains no record of the complete original environment. For example, the logistic script passes `multi_class='multinomial'` explicitly, and the feature-engineering script uses `fillna(method=...)`.

The launcher reads CSVs from `data/raw/` by default. Supply `--data-dir /path/to/csvs` for another location. It executes the selected script in memory with the original data and result paths redirected locally. For the standalone logistic script, it also supplies two missing scikit-learn imports used to print the final classification report and confusion matrix. Model settings, input selection, row order, splits, and random state follow the selected source. Most scripts print results to standard output. The statistics script writes `statistical_analysis_results.csv` to the output directory.

## Classical classifiers

The standalone classical models use shuffled, row-wise 10-fold `KFold` with split seed 42. Most concatenate the CSVs in ascending participant order, take a trailing five-row mean on each of the six accelerometer axes, then remove the first four rows once. This retains the original behavior at participant boundaries. They report accuracy and macro-averaged precision, recall and F1 on each fold and on the pooled out-of-fold predictions. Folds may contain records from the same person in training and testing. The features used by these scripts are accelerometer readings after smoothing, with the SVM exception below.

| Standalone file | Participants used for fitting | Estimator in source |
| --- | --- | --- |
| [har70RF.py](scripts/classical/har70RF.py) | 501–502 | Random forest, 6 trees, Gini impurity. The estimator has no fixed seed, so reruns can differ. |
| [har70DT.py](scripts/classical/har70DT.py) | 501–502 | Decision tree, Gini impurity, best split. No estimator seed is set. |
| [har70logistic.py](scripts/classical/har70logistic.py) | 501–518 | Multinomial logistic regression with the SAG solver and a maximum of 1,000 iterations. |
| [har70NB.py](scripts/classical/har70NB.py) | 501–502 | Gaussian naïve Bayes with library defaults. |
| [har70KNN.py](scripts/classical/har70KNN.py) | 501–502 | K-nearest neighbors with 10 neighbors. |
| [har70LDA.py](scripts/classical/har70LDA.py) | 501–502 | Linear discriminant analysis with library defaults. |
| [har70QDA.py](scripts/classical/har70QDA.py) | 501–502 | Quadratic discriminant analysis with library defaults. |
| [har70hist.py](scripts/classical/har70hist.py) | 501–502 | Histogram gradient boosting, 100 maximum iterations, estimator seed 42. |
| [har70SVM.py](scripts/classical/har70SVM.py) | 502 | SVC with a linear kernel and C = 1. The loader smooths merged records from 501–502, but the fit reads six raw axes from the last file loaded. |

## Neural classifiers

Each script applies the five-row mean first, uses a 70/30 shuffled row split, then trains with Adam and categorical cross-entropy. Each CNN input is a single observation organized by sensor and axis. The recurrent inputs also contain one time step per observation.

| Standalone file | Participants | Architecture and training |
| --- | --- | --- |
| [har70CNN.py](scripts/neural/har70CNN.py) | 501–502 | Reshape to 2 × 3 × 1, zero padding, 32 convolution filters of size 2 × 2, 1 × 1 max pooling, flatten, dense 64, softmax. Split seed 15, 10 epochs, batch size 64. Uses the test split for validation during training. |
| [har70LSTM.py](scripts/neural/har70LSTM.py) | 501–502 | Two LSTM layers with 128 and 64 units, each followed by 50% dropout, then softmax. LabelEncoder maps class IDs before one-hot encoding. Split seed 42, up to 10 epochs, batch size 64, 20% of training rows for validation, early stopping on validation loss with patience 5. |
| [CNN-LSTM.py](scripts/neural/CNN-LSTM.py) | 501 | Time-distributed padding, Conv2D (32 filters), 1 × 1 pooling and flatten, followed by LSTM layers (128 and 64 units), 50% dropout after each, and softmax. The convolution and output layer use **L1** regularization at 0.001. Split seed 42, 10 epochs, batch size 64, test split used for validation. |
| [har70MLP.py](scripts/neural/har70MLP.py) | 501–502 | Two 64-unit ReLU layers and softmax, split seed 42, 100 epochs, batch size 32, 20% training validation. This standalone version retains the string timestamp in its model inputs and cannot train directly on the supplied CSV format. The [neural-models notebook](notebooks/neural-models.ipynb) has a separate MLP variant which drops the timestamp. |

## Preparation and exploration

These files capture separate analyses. In particular, the 25-row feature table in `feaeng.py` is a separate experiment from the standalone six-axis random forest.

| Source | Scope | What it does |
| --- | --- | --- |
| [filitered.py](scripts/preprocessing/filitered.py) | 501–518 | Concatenates all 18 participant CSVs, prints the raw frame, and computes the five-row trailing mean in memory. |
| [stat descri.py](scripts/preprocessing/stat%20descri.py) | 501–518 | Computes per-label descriptive statistics after smoothing, prints the transposed table and writes the original table to CSV. |
| [feaeng.py](scripts/preprocessing/feaeng.py) | 501 | Fills missing values forward then backward and uses non-overlapping complete windows of 25 rows. Calculates 10 summary features and six normalized absolute DC coefficients, labels each window by its first row, and prints the first five windows. |
| [har70outlier.py](scripts/preprocessing/har70outlier.py) | 501–517 | Produces a box plot of numeric features. The final column selection drops the label. |
| [har70noise.py](scripts/exploratory/har70noise.py) | 504 and a planned 512 analysis | Explores back and thigh acceleration magnitudes and their five-row means. The first plot uses a fixed-length x-axis with per-label data of different lengths. Later code has an empty input path and a reference to an undefined merged frame. Treat it as an exploratory source. |
| [har70total.py](scripts/exploratory/har70total.py) | Starts with 501–502, with additional participant-specific sections | Combined working script with missing-value checks, magnitude plots, correlation and VIF analysis, classical classifiers and neural variants. A later input path is empty. Some later models retain the timestamp, and its LSTM section uses raw 504 data standardized before splitting. Use the individually identified scripts or notebook sections for targeted runs. |

The [formula notes](notes/metric-formulas-zh.docx) and notebook outputs provide additional context for interpreting the original analyses. The [source inventory](source-manifest.tsv) maps every original code file and drafting document to its full archive name and SHA-256 digest. The [data manifest](../data/manifest.csv) provides separate digests for the 18 recordings.
