# Breast Cancer Classification Project

This repository contains a small data‑science project that uses the
Wisconsin Breast Cancer (Diagnostic) dataset to build and evaluate
supervised machine‑learning models for tumour classification.  The goal
is to predict whether a breast tumour is **malignant** or **benign**
based on 30 numeric features derived from cell nuclei in digitised
images【287038991818988†L676-L703】.  The dataset consists of 569
samples split into 212 malignant and 357 benign cases【287038991818988†L676-L703】.

> **Why this dataset?**  The Breast Cancer Wisconsin (Diagnostic)
> dataset is a well‑known benchmark that is small enough to train
> several models quickly yet complex enough to showcase key elements
> of a modern machine‑learning workflow.  It comes bundled with
> `scikit‑learn`, so no external downloads are required.

## Dataset

The dataset used in this project originates from the University of
Wisconsin Hospitals, Madison.  Each record describes a fine
needle aspirate (FNA) of a breast mass.  Thirty real‑valued
features characterise the shape, texture and geometry of cell nuclei in
the digitised images【136898347949267†L94-L101】.  Examples include
**radius**, **texture**, **perimeter**, **area**, **smoothness** and
**concavity** of the nuclei.  There are **569 samples**, with **212
malignant** and **357 benign** tumours【287038991818988†L676-L703】.  All features are positive
real numbers and there are no missing values【287038991818988†L676-L703】.

For more details see the [UCI Machine Learning Repository
entry](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
and the [`load_breast_cancer` API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

## Project structure

```text
portfolio_project/
├── README.md                ← Project overview and instructions (this file)
├── requirements.txt         ← Python dependencies
├── src/                     ← Source code
│   ├── train.py             ← Trains several models and saves the best one
│   ├── evaluate.py          ← Evaluates a saved model on a holdout test set
│   └── predict.py           ← Makes predictions on new data using a saved model
├── models/                  ← Saved machine‑learning models (created after training)
├── results/                 ← Metrics, plots and CSV summaries (created after training)
└── notebooks/               ← (Optional) Jupyter notebooks for exploratory analysis
```

## Installation

It is recommended to create a dedicated virtual environment.  You can
install all required packages using `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

To train the models, run the following command from the root of the
repository:

```bash
python src/train.py
```

The script performs the following steps:

1. Loads the dataset into a pandas `DataFrame` and splits it into
   stratified training and test sets (80 %/20 %).
2. Defines three model pipelines: **Logistic Regression**, **Random
   Forest** and **Support Vector Machine**.  Each pipeline includes a
   `StandardScaler` followed by the classifier.  Hyperparameters are
   tuned via a grid search with five‑fold cross‑validation.
3. Evaluates each model on the holdout test set and records metrics such
   as accuracy, precision, recall, F1 score and ROC AUC.  A confusion
   matrix and classification report are saved to the `results/`
   directory.
4. Saves the cross‑validation results to `results/model_comparison.csv`
   and selects the best model based on test accuracy.  The best model
   is persisted to `models/best_model.joblib` using `joblib`.

After training, you can inspect the generated files in the `results/`
directory.  For example, `model_comparison.csv` summarises the
performance of all models and `*_confusion_matrix.png` plots the
confusion matrix.

## Evaluation

To re‑evaluate a saved model (useful after retraining or to verify
performance), run:

```bash
python src/evaluate.py
```

This script loads the model from `models/best_model.joblib`, splits the
dataset with the same random seed used during training, computes a
classification report and confusion matrix, and writes the results to
`results/evaluation/`.

## Making predictions

The `src/predict.py` script applies the trained model to new samples.  It
expects a CSV file with feature columns matching those of the original
dataset.  For example, given a file `my_samples.csv` containing one or
more rows of 30 numeric features:

```bash
python src/predict.py --input my_samples.csv --output my_predictions.csv
```

The script will write a CSV file with columns for the predicted
diagnosis (`malignant` or `benign`) and the associated class
probabilities.  If no `--output` is supplied, the predictions are
printed to the console.

## Notebook

The `notebooks/` directory is left empty for you to explore the
dataset interactively.  You can create a Jupyter notebook and perform
exploratory data analysis, visualise feature distributions or attempt
alternative modelling strategies.  To start a notebook server, install
JupyterLab and run:

```bash
pip install jupyterlab
jupyter lab
```

## References

* **UCI Machine Learning Repository.** *Breast Cancer Wisconsin (Diagnostic)*
  dataset information: features are computed from digitised FNA images
  describing characteristics of cell nuclei【136898347949267†L94-L101】.
* **scikit‑learn documentation.** The dataset contains 569 samples with
  30 real‑valued features; there are 212 malignant and 357 benign
  tumours and no missing values【287038991818988†L676-L703】.

## License

This project is provided for educational purposes.  The code is licensed
under the MIT license.  See `LICENSE` (not included) for details.