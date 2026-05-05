from __future__ import annotations

import numpy as np
import pandas as pd

from auditor._helpers import compute_ece, threshold_status
from auditor.base import AuditFinding, Check
from auditor.constants import (
    ART14_ABSTENTION_THRESHOLD,
    ART14_BRIER_FAIL,
    ART14_BRIER_WARN,
    ART14_ECE_FAIL,
    ART14_ECE_WARN,
    ART14_LOW_CONF_THRESHOLD,
)


def audit(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    protected_attrs: dict[str, pd.Series],
) -> AuditFinding:
    checks: list[Check] = []
    evidence: dict = {}

    y_arr = y_test.to_numpy().astype(float)
    proba = model.predict_proba(X_test)   # (n, 2)
    pos_prob = proba[:, 1]
    max_prob = proba.max(axis=1)

    evidence["y_true"] = y_arr.tolist()
    evidence["y_prob"] = pos_prob.tolist()

    # --- Brier score ---
    brier = float(np.mean((pos_prob - y_arr) ** 2))
    evidence["brier_score"] = round(brier, 6)
    checks.append(Check(
        name="brier_score",
        status=threshold_status(brier, ART14_BRIER_WARN, ART14_BRIER_FAIL),
        value=round(brier, 4),
        threshold=ART14_BRIER_WARN,
        message=f"Brier score: {brier:.4f}",
    ))

    # --- Expected Calibration Error ---
    ece = compute_ece(y_arr, pos_prob)
    evidence["ece"] = round(ece, 6)
    checks.append(Check(
        name="calibration_ece",
        status=threshold_status(ece, ART14_ECE_WARN, ART14_ECE_FAIL),
        value=round(ece, 4),
        threshold=ART14_ECE_WARN,
        message=f"Expected Calibration Error: {ece:.4f}",
    ))

    # --- Low-confidence flag rate (informational only) ---
    low_conf_rate = float((max_prob < ART14_LOW_CONF_THRESHOLD).mean())
    evidence["low_conf_rate"] = round(low_conf_rate, 4)
    evidence["low_conf_threshold"] = ART14_LOW_CONF_THRESHOLD

    # --- Abstention rate (informational only) ---
    abstention_rate = float((max_prob < ART14_ABSTENTION_THRESHOLD).mean())
    evidence["abstention_rate"] = round(abstention_rate, 4)
    evidence["abstention_threshold"] = ART14_ABSTENTION_THRESHOLD

    return AuditFinding(article="14", checks=checks, evidence=evidence)
