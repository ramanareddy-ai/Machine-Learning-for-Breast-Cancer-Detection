"""
predict.py
----------

This script loads a pre‑trained classifier (saved by ``train.py``) and applies
it to new samples.  Users must provide a path to a CSV file containing
feature values in the same order as the breast cancer dataset.  The script
outputs predicted class labels (``malignant`` or ``benign``) and class
probabilities to either a new CSV file or standard output.

Usage
-----

To predict the diagnosis for new samples stored in ``my_samples.csv`` and
write the results to ``predictions.csv``, run:

.. code-block:: bash

   python -m src.predict --input my_samples.csv --output predictions.csv

If no ``--output`` is specified, predictions will be printed to the console.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: Path):
    """Load the persisted model from disk."""
    return joblib.load(model_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict breast cancer diagnosis for new samples.")
    parser.add_argument("--input", required=True, help="Path to CSV file containing feature values.")
    parser.add_argument("--output", help="Optional path to write predictions as CSV.")
    parser.add_argument("--model", default="models/best_model.joblib", help="Path to the trained model.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.input)
    model_path = Path(args.model)
    # Load new data
    try:
        X_new = pd.read_csv(input_path)
    except Exception as e:
        raise SystemExit(f"Error reading input file {input_path}: {e}")
    # Load model
    model = load_model(model_path)
    # Make predictions
    preds = model.predict(X_new)
    # Try to get probabilities if available
    try:
        probas = model.predict_proba(X_new)
    except Exception:
        # Some estimators do not implement predict_proba
        probas = None
    # Map numeric labels to strings
    label_map = {0: "malignant", 1: "benign"}
    class_labels = [label_map.get(int(p), str(p)) for p in preds]
    # Build result DataFrame
    output_df = pd.DataFrame({
        "prediction": class_labels,
    })
    if probas is not None:
        output_df["prob_malignant"] = probas[:, 0]
        output_df["prob_benign"] = probas[:, 1]
    # Write results
    if args.output:
        out_path = Path(args.output)
        output_df.to_csv(out_path, index=False)
        print(f"Predictions written to {out_path}")
    else:
        print(output_df)


if __name__ == "__main__":
    main()