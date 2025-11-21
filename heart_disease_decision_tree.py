"""
Train a Decision Tree model to predict heart disease risk levels.

Usage:
    python heart_disease_decision_tree.py \
        --data-path heart_disease_cleaned.csv \
        --output-dir outputs
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heart disease risk prediction with a Decision Tree."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("heart_disease_cleaned.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where metrics and artifacts will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset reserved for evaluation.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth of the decision tree.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=10,
        help="Minimum samples per leaf node.",
    )
    return parser.parse_args()


def map_risk_level(num_value: int) -> str:
    """Convert the original `num` severity to a risk bucket."""
    if num_value <= 0:
        return "low"
    if num_value <= 2:
        return "moderate"
    return "high"


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create an OneHotEncoder that works across sklearn versions."""
    signature = inspect.signature(OneHotEncoder)
    params = signature.parameters
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in params:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def build_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str],
    max_depth: int,
    min_samples_leaf: int,
) -> Pipeline:
    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                _make_one_hot_encoder(),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))

    if not transformers:
        raise ValueError("No features available to train the model.")

    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def get_feature_importances(
    model: Pipeline, categorical_cols: List[str], numeric_cols: List[str]
) -> List[Dict[str, float]]:
    classifier: DecisionTreeClassifier = model.named_steps["classifier"]
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]

    feature_names: List[str] = []
    if categorical_cols:
        ohe: OneHotEncoder = preprocessor.named_transformers_["categorical"]
        feature_names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
    if numeric_cols:
        feature_names.extend(numeric_cols)

    importances = classifier.feature_importances_
    feature_importances = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(feature_names, importances)
    ]
    feature_importances.sort(key=lambda item: item["importance"], reverse=True)
    return feature_importances


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)

    if "num" not in df.columns:
        raise ValueError(
            "Dataset must contain the 'num' column describing disease severity."
        )

    df = df.copy()
    df["risk_level"] = df["num"].apply(map_risk_level)

    drop_cols = {"id", "num", "risk_level"}
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df["risk_level"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    model = build_pipeline(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=["low", "moderate", "high"])
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validated accuracy using the full dataset
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    feature_importances = get_feature_importances(
        model, categorical_cols, numeric_cols
    )

    metrics = {
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["low", "moderate", "high"],
            "matrix": cm.tolist(),
        },
        "cross_val_accuracy_mean": float(np.mean(cv_scores)),
        "cross_val_accuracy_std": float(np.std(cv_scores)),
        "feature_importances": feature_importances,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    feature_df = pd.DataFrame(feature_importances)
    feature_df.to_csv(args.output_dir / "feature_importances.csv", index=False)

    print("Decision Tree Heart Disease Risk Prediction")
    print(f"Dataset: {args.data_path}")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(
        "Cross-validation Accuracy: "
        f"{np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}"
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=["low", "moderate", "high"], columns=["low", "moderate", "high"]))
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

