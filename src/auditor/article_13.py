from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import shap

from auditor.base import AuditFinding, AuditStatus, Check
from auditor.constants import ART13_MODEL_CARD_FIELDS

_MAX_SHAP_SAMPLES = 200


def _shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]  # binary: take positive-class array
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv, None
    except Exception as e:
        return None, str(e)


def _model_card(model, X: pd.DataFrame, protected_attrs: dict) -> dict:
    return {
        "model_type": type(model).__name__,
        "training_date": datetime.date.today().isoformat(),
        "feature_list": list(X.columns),
        "target": "hired",
        "intended_use": (
            "CV/resume screening assistance for employment decisions "
            "(Annex III §4 — employment, workers management)."
        ),
        "limitations": (
            "May encode historical hiring biases present in training data. "
            "Human review is required before any adverse employment decision."
        ),
        "protected_attrs_documented": list(protected_attrs.keys()),
    }


def audit(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    protected_attrs: dict[str, pd.Series],
) -> AuditFinding:
    checks: list[Check] = []
    evidence: dict = {}

    sample = X_test.iloc[: min(_MAX_SHAP_SAMPLES, len(X_test))]
    sv, err = _shap_values(model, sample)

    if err is not None:
        checks.append(Check(
            name="shap_global_importance",
            status=AuditStatus.FAIL,
            value="error",
            threshold=None,
            message=f"SHAP computation failed: {err}",
        ))
    else:
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:5]
        top5 = {sample.columns[i]: round(float(mean_abs[i]), 6) for i in top_idx}

        evidence["shap_global_importance"] = top5
        evidence["shap_mean_abs"] = {
            col: round(float(v), 6) for col, v in zip(sample.columns, mean_abs)
        }
        evidence["shap_local"] = {
            f"sample_{i}": {
                col: round(float(sv[i, j]), 6)
                for j, col in enumerate(sample.columns)
            }
            for i in range(min(5, len(sample)))
        }
        checks.append(Check(
            name="shap_global_importance",
            status=AuditStatus.PASS,
            value=next(iter(top5)),
            threshold=None,
            message=f"SHAP computed. Top 5 features: {list(top5)}",
        ))

    card = _model_card(model, X_test, protected_attrs)
    evidence["model_card"] = card
    empty = [f for f in ART13_MODEL_CARD_FIELDS if not card.get(f)]
    checks.append(Check(
        name="model_card_completeness",
        status=AuditStatus.WARN if empty else AuditStatus.PASS,
        value=f"{len(ART13_MODEL_CARD_FIELDS) - len(empty)}/{len(ART13_MODEL_CARD_FIELDS)}",
        threshold="all fields non-empty",
        message="Model card complete." if not empty else f"Empty fields: {empty}",
    ))

    return AuditFinding(article="13", checks=checks, evidence=evidence)
