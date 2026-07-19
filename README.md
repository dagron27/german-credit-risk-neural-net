# German Credit Risk Neural Network Classifier

![CI](https://github.com/dagron27/german-credit-risk-neural-net/actions/workflows/ci.yml/badge.svg)

**Course:** `CSCI 441/541, Neural Networks, Fall 2025`

**Assignment:** `csci-441-541-project3`

This repository originally lived under a GitHub Classroom organization (`github.com/St-Cloud-State/...`); it has since been moved to this personal GitHub account (`github.com/dagron27/german-credit-risk-neural-net`). The Classroom copy remains the official course submission record and was left untouched.

## Assignment Intent

The assignment asked students to rebuild the Project 2 classifier as a deep-learning model: train an MLP with Keras, visualize training with TensorBoard, and tune hyperparameters with Keras Tuner, following the textbook's neural-network hyperparameter guidance (pages 349-353, Table 10-2) and its example learning-curve and TensorBoard figures (10-11, 10-16).

**Confirmed implemented**, per the notebook (see Overview below): data preparation, a Sequential Keras MLP with the requested layer/dropout/batch-norm structure, `model.summary()`-style display of each layer's name, output shape, and parameter counts, training, evaluation with accuracy/loss learning curves plotted from `history.history` (matching Figure 10-11) plus a second TensorBoard-based visualization pass (matching Figure 10-16), prediction, and a Keras Tuner `BayesianOptimization` hyperparameter search reporting the best configuration and accuracy.

**Partially met -- two specific gaps confirmed.** The assignment's hyperparameter-tuning step asks for tuning "with all the bells and whistles -- save checkpoints, use early stopping, and plot learning curves using TensorBoard." Checked directly against the tuner's callback list: `EarlyStopping` and `TensorBoard` are both present, but there is no `ModelCheckpoint` (or equivalent) anywhere in the notebook -- no checkpoints are saved during the tuning search. Separately, the assignment's final component is a written "Conclusion"; the notebook has no dedicated Conclusion heading or closing analysis paragraph -- its last section is "Final Model," which reports the final metrics but doesn't discuss them as a conclusion.

## Overview

This project trains a multilayer perceptron (MLP) with Keras/TensorFlow to classify loan applicants in the [German Credit Risk dataset](https://www.openml.org/data/get_csv/31/dataset_31_german_credit.csv) (1,000 samples, 21 attributes, downloaded from OpenML) as `good` or `bad` credit risk. It is a rebuild of an earlier (Project 2) classifier using a deep-learning approach instead of classical ML models.

The notebook `notebooks/proj3.ipynb` (107 cells) walks through:

1. **Data acquisition** — downloads the CSV from OpenML on first run if `data/german_credit_data.csv` is not already present (it is present in this repo), otherwise loads it locally.
2. **Exploration** — shape/info/describe, categorical value counts, a stratified 70/30 train/test split on the `class` label, scatter matrix and correlation heatmap analysis, and an exploratory (ultimately unused — see Known Issues) feature-engineering pass.
3. **Preprocessing pipeline** — an sklearn `ColumnTransformer` combining median imputation, standard scaling, log1p transforms, ordinal + one-hot encoding for categoricals, and several custom ratio features (e.g., `credit_amount_per_existing_credits`), applied via `IsolationForest`-based outlier removal (contamination=0.12) beforehand.
4. **Baseline model** — a `Sequential` Keras MLP: Dense(100, relu) → BatchNorm → Dropout(0.3) → Dense(50, relu) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid), trained 30 epochs with Adam, evaluated with accuracy/precision/recall/F1/ROC-AUC, a confusion matrix, and an ROC curve.
5. **TensorBoard integration** — standard `TensorBoard` callback plus a custom `DetailedTensorBoardCallback` that writes per-epoch scalar metrics, retraining the model to produce the logs (written under `logs/`, which is git-ignored).
6. **Hyperparameter tuning** — Keras Tuner `BayesianOptimization` (30 trials) searching input/hidden layer widths, dropout rates, and learning rate, with `EarlyStopping` and TensorBoard logging (trial artifacts written under `credit_tuning/`, also git-ignored).
7. **Final model** — the best hyperparameter configuration (192 input units, 0.30 input dropout, 16 hidden units, 0.50 hidden dropout, learning rate ≈ 7.36e-4) is retrained and re-evaluated, achieving a best validation accuracy of **0.8226**, test-set precision **0.7830**, recall **0.8762**, F1 **0.8270**, ROC-AUC **0.7295**. The final model is saved as `models/best_german_credit_model.h5` (the committed file in this repo).

## Dependencies

A `requirements.txt` is now included (`pip install -r requirements.txt`).
Based on the notebook's imports, the following are required:

- Python >= 3.7 (kernel metadata specifies a `.venv` running Python 3.10; notebook asserts `sys.version_info >= (3, 7)`)
- `pandas`
- `numpy`
- `scikit-learn` >= 1.0.1 (notebook asserts this via `packaging.version`)
- `tensorflow` (provides `tensorflow.keras`)
- `tensorboard` -- not imported directly, but required at runtime by
  `tf.keras.callbacks.TensorBoard`/`tf.summary.scalar` (used in the
  baseline-model and tuning `model.fit()` calls) and by the
  `%load_ext tensorboard`/`%tensorboard` magics; omitting it raises
  `TBNotInstalledError` from inside `model.fit()` -- see Known Issues
- `keras-tuner`
- `matplotlib`
- `seaborn`
- `packaging`
- Jupyter/IPython (for the `%load_ext tensorboard` and `%tensorboard` magics)
- `urllib.request` (standard library — used only if the dataset CSV must be re-downloaded)

## Environment Setup

1. Create and activate a virtual environment (the notebook's kernel metadata references `.venv`), e.g.:
   ```
   python -m venv .venv
   .venv\Scripts\activate   (Windows)  /  source .venv/bin/activate   (macOS/Linux)
   ```
2. Install dependencies from the committed manifest:
   ```
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open `notebooks/proj3.ipynb`:
   ```
   jupyter notebook notebooks/proj3.ipynb
   ```
4. `data/german_credit_data.csv` is already committed, so no network access is required to run the notebook end to end. If that file is deleted, cell 12 will attempt to re-download it from `https://www.openml.org/data/get_csv/31/dataset_31_german_credit.csv` — this requires outbound internet access and performs no integrity/hash check on the downloaded content.
5. TensorBoard cells (`%load_ext tensorboard`, `%tensorboard --logdir ...`) only work inside a Jupyter/IPython kernel; running the notebook as a plain `.py` script will fail on those cells.
6. Hyperparameter tuning (`tuner.search`, 30 Bayesian-optimization trials) and TensorBoard logging are computationally the most expensive steps and can take a long time on CPU-only machines.

## Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every `push` (any branch, no filter) and can also be triggered manually via `workflow_dispatch`. It installs `requirements.txt` on `ubuntu-latest` and then re-executes `notebooks/proj3.ipynb` from scratch with `jupyter nbconvert --execute` — this is a full retraining run, including the baseline MLP, the TensorBoard retrain, and the complete 30-trial Keras Tuner `BayesianOptimization` search, not a reduced or mocked version. `max_trials` and the various `epochs` values are hardcoded inline in the notebook rather than exposed as overridable variables, so there is no lightweight way to shrink this run without editing the notebook.

As a result, this CI job takes noticeably longer than a typical notebook CI job — the tensorflow/keras-tuner install alone is heavier than a scikit-learn-only dependency set, and the tuner search adds significant wall-clock time on top of that. This is expected and intentional, not a sign of a hung or misconfigured job; the workflow sets a generous `timeout-minutes` purely as a safety net against a truly stuck run, not as an indication of normal runtime.

GitHub Actions runners are ephemeral and execution output is written to a non-committed path (not `--inplace`), so a CI run never overwrites the committed `notebooks/proj3.ipynb` or the committed `models/best_german_credit_model.h5`. Similarly, any TensorBoard log files (`logs/`) written while the notebook executes in CI exist only for the lifetime of that runner and do not persist anywhere past the job.

## Known Issues

### Dead Code: Discarded Feature-Engineering Block

In the "Explore the Data" section, cells compute a label-encoded `encoded_class` column and roughly 18 hand-engineered ratio/product features (`age_per_credit_amount`, `age_per_duration`, `credit_amount_per_duration`, etc. — see cell under "Experimenting with Attribute Contributions") on `credit_data`, purely to inspect correlations. Immediately afterward, `credit_data` is reassigned from scratch via `credit_data = strat_train_set.drop("class", axis=1)` (Data Preparation section). This discards every engineered column from the exploration block before the preprocessing `ColumnTransformer` ever runs. The final pipeline defines its own separate, overlapping set of ratio features (via `ratio_pipeline()`), so the earlier block never influences the trained model — it is exploratory analysis left in the notebook, not part of the production feature set.

**Fix-it plan:** Either (a) intentionally wire the exploratory ratio features into the `ColumnTransformer` if they are believed to add value, deduplicating against the pipeline's own ratio features, or (b) leave the block as EDA-only but add a markdown note clarifying that it does not feed the trained model, so future readers/maintainers do not assume those columns are part of the final feature set.

### Unused Variable in Fine-Tuning Section

In the "Fine Tuning" section, `predicted_classes = (predictions > 0.5).astype(int)` is computed but never referenced afterward.

**Fix-it plan:** Remove the unused assignment, or use it in the immediately following accuracy printout instead of recomputing metrics via `model.evaluate`.

### Keras/H5 Model Deserialization Trust Boundary

The repository includes a committed, pre-trained model file, `models/best_german_credit_model.h5`, produced by `best_model.save('../models/best_german_credit_model.h5')`. Keras itself flags this as a legacy format at save time (`WARNING:absl: ... This file format is considered legacy ...`). More importantly from a security standpoint: loading a Keras/H5 model with `tf.keras.models.load_model()` can execute arbitrary Python code if the file contains a `Lambda` layer or other layer with a serialized Python callable, because Keras deserializes those via `pickle`/`marshal`-style mechanisms. This is a well-known class of risk for shared/untrusted `.h5` (and legacy `.pkl`) model files — a malicious `.h5` can achieve code execution simply by being loaded, not just by being "run." In this repository the file is self-produced by the project's own notebook and is presumed trustworthy for that reason alone, but this trust cannot be assumed for any `.h5` file obtained from an external or unverified source.

**Fix-it plan:** For any future work that loads model files from outside this repository (or accepts them from users), do not call `load_model()` directly on untrusted input. Prefer the newer Keras native `.keras` format, load with `safe_mode=True` where supported, inspect the model architecture (e.g., check for `Lambda` layers) before loading, or load model weights only into a known-safe architecture definition instead of deserializing the full model graph. Treat any `.h5`/`.keras`/`.pkl` file from an untrusted source the same as untrusted executable code.

### CI Failure: Missing `tensorboard` Dependency -- Fixed

**Status: resolved.** CI failed with `Process completed with exit code 1`,
raised from inside `model.fit()`. Reproduced locally: the actual error is
`tensorflow.python.summary.tb_summary.TBNotInstalledError: TensorBoard is
not installed, missing implementation for tf.summary.scalar`.
`tf.keras.callbacks.TensorBoard` (used in the baseline-model `model.fit()`
call) calls `tf.summary.scalar` internally, which requires the standalone
`tensorboard` PyPI package -- a separate install from `tensorflow` itself,
and one this repository's `requirements.txt` never listed. Confirmed
locally with `tensorboard` installed that the exact same `model.fit()`
call (with a `TensorBoard` callback) then completes successfully, so this
was not a CI timeout (`timeout-minutes: 30`, per the runtime note above)
as it might first appear from the generic exit-code-1 message -- it's a
missing dependency, reproducible on the very first `model.fit()` call
that uses the callback, well within the timeout.

**Fix applied:** added `tensorboard` to `requirements.txt`.

## Status

This repository is an archival record of a completed course assignment (Project 3, CSCI 441/541) and is not under active development. The notebook runs end to end using the committed dataset and produces a final tuned MLP with best validation accuracy 0.8226 and test-set ROC-AUC 0.7295, saved as `models/best_german_credit_model.h5`. No further feature work is planned; the items above are documented for anyone reusing or extending this code, not as an indication that fixes are in progress.
