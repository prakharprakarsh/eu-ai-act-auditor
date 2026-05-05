from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_hiring_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X, y)
    return clf


if __name__ == "__main__":
    from models.synthetic_data import generate_hiring_dataset

    X, y = generate_hiring_dataset(n=5000, seed=42, bias_strength=0.15)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_hiring_model(X_train, y_train)

    acc = (model.predict(X_test) == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")

    out = Path(__file__).parents[2] / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    X_train.to_parquet(out / "X_train.parquet", index=False)
    X_test.to_parquet(out / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(out / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(out / "y_test.parquet", index=False)

    print(f"Saved model and splits to {out}")
