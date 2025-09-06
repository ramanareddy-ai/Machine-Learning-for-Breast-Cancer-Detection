"""
train.py
---------

This script trains and evaluates a number of machine‑learning models on the
Wisconsin Breast Cancer (Diagnostic) dataset.  The dataset is loaded from
``scikit‑learn`` using :func:`sklearn.datasets.load_breast_cancer`, which
provides 569 samples and 30 numeric features describing characteristics of
cell nuclei extracted from digitised images of fine needle aspiration
samples【287038991818988†L676-L703】.  The goal is to predict whether a
tumour is malignant or benign (two classes: ``malignant`` and ``benign``)
using these features【287038991818988†L676-L703】.  This script trains three
different classifiers—Logistic Regression, Random Forest and Support Vector
Machine—evaluates them using stratified train/test splits and five‑fold
cross‑validation, selects the best performing model and persists it to
disk.  It also exports evaluation metrics and a confusion matrix plot for
inspection.

Usage
-----

From the root of the project repository run the following command in a
terminal to train the models and write outputs into the ``models/`` and
``results/`` directories:

.. code-block:: bash

   python -m src.train

The script does not take any arguments.  When finished it will save the
trained model to ``models/best_model.joblib`` and write a CSV file with
cross‑validation scores to ``results/model_comparison.csv``.  A
classification report and confusion matrix plot are written into
``results/`` as well.
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Wisconsin Breast Cancer dataset into a pandas DataFrame.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame containing the feature columns.
    y : pandas.Series
        Series containing the target values (0 for malignant, 1 for benign).
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = pd.Series(data.target, name="target")
    return X, y


def build_models() -> dict[str, Pipeline]:
    """Construct a dictionary of modelling pipelines.

    Each pipeline includes a :class:`~sklearn.preprocessing.StandardScaler` for
    feature scaling followed by a classifier.  Different hyperparameter grids
    are defined for each model to enable cross‑validated tuning.

    Returns
    -------
    models : dict[str, Pipeline]
        Mapping from a human‑readable model name to a ``Pipeline`` instance.
    params : dict[str, dict]
        Mapping from model name to a hyperparameter grid for
        :class:`~sklearn.model_selection.GridSearchCV`.
    """
    models: dict[str, Pipeline] = {}
    params: dict[str, dict] = {}

    # Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="liblinear")),
    ])
    lr_params = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2", "l1"],
    }
    models["Logistic Regression"] = lr_pipeline
    params["Logistic Regression"] = lr_params

    # Random Forest
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    rf_params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    models["Random Forest"] = rf_pipeline
    params["Random Forest"] = rf_params

    # Support Vector Machine
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True)),
    ])
    svm_params = {
        "clf__C": [0.5, 1.0, 2.0],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    }
    models["Support Vector Machine"] = svm_pipeline
    params["Support Vector Machine"] = svm_params

    return models, params


def evaluate_and_save(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, results_dir: Path) -> dict:
    """Evaluate a fitted model on a holdout test set, save metrics and plots.

    Parameters
    ----------
    model : Pipeline
        A fitted scikit‑learn pipeline.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series
        Test target labels.
    model_name : str
        Human‑readable name of the model for labelling outputs.
    results_dir : pathlib.Path
        Directory in which to save output files.

    Returns
    -------
    metrics : dict
        Dictionary containing scalar evaluation metrics.
    """
    preds = model.predict(X_test)
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
    except Exception:
        roc_auc = np.nan

    # Classification report
    report = classification_report(y_test, preds, target_names=["malignant", "benign"], output_dict=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["malignant", "benign"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    cm_path = results_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    metrics = {
        "accuracy": report["accuracy"],
        "precision_malignant": report["malignant"]["precision"],
        "recall_malignant": report["malignant"]["recall"],
        "f1_malignant": report["malignant"]["f1-score"],
        "precision_benign": report["benign"]["precision"],
        "recall_benign": report["benign"]["recall"],
        "f1_benign": report["benign"]["f1-score"],
        "roc_auc": roc_auc,
    }
    # Save classification report as JSON
    report_path = results_dir / f"{model_name.lower().replace(' ', '_')}_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return metrics


def main() -> None:
    # Create output directories
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    X, y = load_data()
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Get models and parameter grids
    models, param_grids = build_models()

    # Prepare cross‑validation scheme
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Track scores and fitted models
    summary_records = []
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"\nTraining model: {name}")
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        fitted_models[name] = best_estimator

        # Evaluate on test set
        metrics = evaluate_and_save(best_estimator, X_test, y_test, name, results_dir)
        metrics_row = {
            "model": name,
            "best_params": grid.best_params_,
            **metrics,
        }
        summary_records.append(metrics_row)

    # Determine the best model based on accuracy
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(results_dir / "model_comparison.csv", index=False)
    best_idx = summary_df["accuracy"].idxmax()
    best_model_name = summary_df.loc[best_idx, "model"]
    best_model = fitted_models[best_model_name]
    print(f"\nBest model: {best_model_name} with accuracy {summary_df.loc[best_idx, 'accuracy']:.4f}")

    # Persist the best model to disk
    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    # Save metrics summary as JSON
    summary_json_path = results_dir / "model_comparison.json"
    summary_df.to_json(summary_json_path, orient="records", indent=2)
    print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    main()