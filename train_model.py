"""Train PhishLens models from a documented labelled URL CSV.

Expected columns: url,label where label is 0 (legitimate) or 1 (suspicious).
The script never fetches or visits a URL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, extract_features, normalise_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhishLens URL-text classifiers.")
    parser.add_argument("dataset", type=Path, help="CSV containing url,label columns")
    parser.add_argument("--output", type=Path, default=Path("models/model_bundle.joblib"))
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    required = {"url", "label"}
    if missing := required - set(data.columns):
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    data = data[["url", "label"]].dropna().copy()
    data["url"] = data.url.astype(str).map(normalise_url)
    data["label"] = pd.to_numeric(data.label, errors="raise").astype(int)
    if not data.label.isin([0, 1]).all():
        raise ValueError("label must contain only 0 (legitimate) and 1 (suspicious).")
    data = data.drop_duplicates(subset="url").reset_index(drop=True)
    if data.label.nunique() != 2 or data.label.value_counts().min() < 4:
        raise ValueError("Need at least four unique URLs in each class after deduplication.")

    X = pd.DataFrame([extract_features(url) for url in data.url], columns=FEATURE_NAMES)
    y = data.label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    forest = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    )
    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    print(f"Unique URLs after cleaning: {len(data)}")
    print("\nLogistic regression (held-out test set)")
    print(classification_report(y_test, logistic.predict(X_test), target_names=["legitimate", "suspicious"], zero_division=0))
    print("Random forest (held-out test set)")
    print(classification_report(y_test, forest.predict(X_test), target_names=["legitimate", "suspicious"], zero_division=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"logistic": logistic, "forest": forest, "X_train": X_train, "X_test": X_test,
         "y_train": y_train, "y_test": y_test, "source": str(args.dataset), "row_count": len(data)},
        args.output,
    )
    print(f"Saved model bundle: {args.output}")


if __name__ == "__main__":
    main()
