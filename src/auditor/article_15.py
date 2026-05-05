from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from auditor._helpers import threshold_status
from auditor.base import AuditFinding, Check
from auditor.constants import (
    ART15_ACCURACY_FAIL,
    ART15_ACCURACY_WARN,
    ART15_FAIRNESS_FAIL,
    ART15_FAIRNESS_WARN,
    ART15_NOISE_SIGMA,
    ART15_ROBUSTNESS_FAIL,
    ART15_ROBUSTNESS_WARN,
)


def audit(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    protected_attrs: dict[str, pd.Series],
) -> AuditFinding:
    checks: list[Check] = []
    evidence: dict = {}

    y_arr = y_test.to_numpy()
    y_pred = model.predict(X_test)

    # --- Performance metrics ---
    acc = float(accuracy_score(y_arr, y_pred))
    evidence["accuracy"] = round(acc, 4)
    evidence["precision"] = round(float(precision_score(y_arr, y_pred, zero_division=0)), 4)
    evidence["recall"] = round(float(recall_score(y_arr, y_pred, zero_division=0)), 4)
    evidence["f1"] = round(float(f1_score(y_arr, y_pred, zero_division=0)), 4)
    checks.append(Check(
        name="accuracy",
        status=threshold_status(acc, ART15_ACCURACY_WARN, ART15_ACCURACY_FAIL, higher_is_worse=False),
        value=round(acc, 4),
        threshold=ART15_ACCURACY_WARN,
        message=f"Accuracy: {acc:.4f}, F1: {evidence['f1']:.4f}",
    ))

    # --- Fairness: demographic parity and equalized odds per protected attr ---
    dpd_results: dict = {}
    eod_results: dict = {}
    for attr, series in protected_attrs.items():
        sf = series.to_numpy()

        dpd = float(demographic_parity_difference(y_arr, y_pred, sensitive_features=sf))
        dpd_results[attr] = round(dpd, 4)
        checks.append(Check(
            name=f"demographic_parity_{attr}",
            status=threshold_status(abs(dpd), ART15_FAIRNESS_WARN, ART15_FAIRNESS_FAIL),
            value=round(dpd, 4),
            threshold=ART15_FAIRNESS_WARN,
            message=f"Demographic parity diff ({attr}): {dpd:.4f}",
        ))

        eod = float(equalized_odds_difference(y_arr, y_pred, sensitive_features=sf))
        eod_results[attr] = round(eod, 4)
        checks.append(Check(
            name=f"equalized_odds_{attr}",
            status=threshold_status(abs(eod), ART15_FAIRNESS_WARN, ART15_FAIRNESS_FAIL),
            value=round(eod, 4),
            threshold=ART15_FAIRNESS_WARN,
            message=f"Equalized odds diff ({attr}): {eod:.4f}",
        ))

    evidence["demographic_parity"] = dpd_results
    evidence["equalized_odds"] = eod_results

    # --- Robustness: Gaussian noise on continuous (float) features ---
    rng = np.random.default_rng(42)
    float_cols = X_test.select_dtypes(include="float").columns
    X_noisy = X_test.copy()
    noise = rng.normal(0.0, ART15_NOISE_SIGMA, size=(len(X_test), len(float_cols)))
    X_noisy[float_cols] = X_noisy[float_cols].to_numpy() + noise

    preds_noisy = model.predict(X_noisy)
    stability = float((y_pred == preds_noisy).mean())
    evidence["robustness_stability"] = round(stability, 4)
    evidence["noise_sigma"] = ART15_NOISE_SIGMA
    checks.append(Check(
        name="robustness",
        status=threshold_status(stability, ART15_ROBUSTNESS_WARN, ART15_ROBUSTNESS_FAIL, higher_is_worse=False),
        value=round(stability, 4),
        threshold=ART15_ROBUSTNESS_WARN,
        message=f"Prediction stability under Gaussian noise (σ={ART15_NOISE_SIGMA}): {stability:.4f}",
    ))

    return AuditFinding(article="15", checks=checks, evidence=evidence)
