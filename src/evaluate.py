"""
evaluate.py
------------

Standalone script for evaluating a pre‑trained model on the Wisconsin Breast
Cancer (Diagnostic) dataset.  It loads the model saved by ``train.py``
from the ``models/`` directory, splits the dataset into a stratified
train/test split (the same random seed as used during training) and
computes various performance metrics.  Results, including the
classification report and confusion matrix image, are written to the
``results/`` directory.  If you wish to evaluate a different model,
replace ``best_model.joblib`` with the appropriate filename.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = pd.Series(data.target, name="target")
    return X, y


def evaluate_model(model_path: Path, results_dir: Path) -> None:
    X, y = load_data()
    # Use the same test size and random seed as in training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
    except Exception:
        roc_auc = float("nan")
    report = classification_report(
        y_test, preds, target_names=["malignant", "benign"], output_dict=True
    )
    # Save report
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["malignant", "benign"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix: Evaluation")
    fig.tight_layout()
    cm_path = results_dir / "evaluation_confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    # Write summary metrics to a text file
    summary = {
        "accuracy": report["accuracy"],
        "roc_auc": roc_auc,
    }
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation complete. Metrics saved to {results_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "best_model.joblib"
    results_dir = project_root / "results" / "evaluation"
    evaluate_model(model_path, results_dir)