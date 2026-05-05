import numpy as np
import pytest

from models.synthetic_data import generate_hiring_dataset
from models.hiring_classifier import train_hiring_model


def test_model_trains_and_predicts(trained_hiring_model, hiring_data):
    X, _ = hiring_data
    preds = trained_hiring_model.predict(X)
    assert set(preds).issubset({0, 1})


def test_model_accuracy_above_threshold(trained_hiring_model, hiring_data):
    X, y = hiring_data
    acc = (trained_hiring_model.predict(X) == y).mean()
    assert acc > 0.70, f"Accuracy {acc:.3f} is below 0.70"


def test_gender_bias_present():
    X, _ = generate_hiring_dataset(n=5000, seed=42, bias_strength=0.15)
    _, y = generate_hiring_dataset(n=5000, seed=42, bias_strength=0.15)
    model = train_hiring_model(X, y)

    preds = model.predict(X)
    rate_gender0 = preds[X["gender"] == 0].mean()
    rate_gender1 = preds[X["gender"] == 1].mean()

    assert rate_gender0 > rate_gender1, (
        f"Expected gender=0 hire rate ({rate_gender0:.3f}) > gender=1 ({rate_gender1:.3f})"
    )
    assert (rate_gender0 - rate_gender1) > 0.03, (
        f"Bias margin too small: {rate_gender0 - rate_gender1:.3f}"
    )


def test_age_bias_present():
    X, _ = generate_hiring_dataset(n=5000, seed=42, bias_strength=0.15)
    _, y = generate_hiring_dataset(n=5000, seed=42, bias_strength=0.15)
    model = train_hiring_model(X, y)

    preds = model.predict(X)
    rate_young = preds[X["age_band"] == 0].mean()
    rate_old = preds[X["age_band"] == 2].mean()

    assert rate_young > rate_old, (
        f"Expected age_band=0 hire rate ({rate_young:.3f}) > age_band=2 ({rate_old:.3f})"
    )


def test_reproducibility():
    X1, y1 = generate_hiring_dataset(n=200, seed=7)
    X2, y2 = generate_hiring_dataset(n=200, seed=7)
    assert (X1 == X2).all().all()
    assert (y1 == y2).all()


def test_feature_columns():
    X, y = generate_hiring_dataset(n=100)
    expected = {
        "years_experience", "education_level", "prior_role_match",
        "skills_overlap", "gap_months", "gender", "age_band", "ethnicity_proxy",
    }
    assert set(X.columns) == expected
    assert y.name == "hired"
