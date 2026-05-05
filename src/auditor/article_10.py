from __future__ import annotations

import pandas as pd
from scipy import stats

from auditor._helpers import threshold_status
from auditor.base import AuditFinding, Check
from auditor.constants import (
    ART10_CLASS_BALANCE_FAIL,
    ART10_CLASS_BALANCE_WARN,
    ART10_COVERAGE_FAIL,
    ART10_COVERAGE_WARN,
    ART10_KS_PVALUE_FAIL,
    ART10_KS_PVALUE_WARN,
    ART10_MISSING_FAIL,
    ART10_MISSING_WARN,
)


def audit(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    protected_attrs: dict[str, pd.Series],
    X_train: pd.DataFrame | None = None,
) -> AuditFinding:
    checks: list[Check] = []
    evidence: dict = {}

    # --- Class balance ---
    fracs = y_test.value_counts(normalize=True)
    min_frac = float(fracs.min())
    evidence["class_distribution"] = {int(k): float(v) for k, v in fracs.items()}
    checks.append(Check(
        name="class_balance",
        status=threshold_status(min_frac, ART10_CLASS_BALANCE_WARN, ART10_CLASS_BALANCE_FAIL, higher_is_worse=False),
        value=round(min_frac, 4),
        threshold=ART10_CLASS_BALANCE_WARN,
        message=f"Minority class fraction: {min_frac:.3f}",
    ))

    # --- Protected attribute coverage ---
    coverage: dict = {}
    for attr, series in protected_attrs.items():
        group_fracs = series.value_counts(normalize=True)
        min_g = float(group_fracs.min())
        coverage[attr] = {int(k): float(v) for k, v in group_fracs.items()}
        checks.append(Check(
            name=f"protected_coverage_{attr}",
            status=threshold_status(min_g, ART10_COVERAGE_WARN, ART10_COVERAGE_FAIL, higher_is_worse=False),
            value=round(min_g, 4),
            threshold=ART10_COVERAGE_WARN,
            message=f"{attr} min group fraction: {min_g:.3f}",
        ))
    evidence["protected_coverage"] = coverage

    # --- Missing values ---
    miss = X_test.isnull().mean()
    evidence["missing_rates"] = {col: float(v) for col, v in miss.items()}
    max_miss = float(miss.max())
    worst_col = str(miss.idxmax())
    checks.append(Check(
        name="missing_values",
        status=threshold_status(max_miss, ART10_MISSING_WARN, ART10_MISSING_FAIL),
        value=round(max_miss, 4),
        threshold=ART10_MISSING_WARN,
        message=f"Max missing rate {max_miss:.3f} (col: {worst_col})",
    ))

    # --- KS drift (optional — skipped if X_train not provided) ---
    if X_train is not None:
        num_cols = X_test.select_dtypes(include="number").columns
        ks: dict = {}
        min_p, worst_ks_col = 1.0, ""
        for col in num_cols:
            _, p = stats.ks_2samp(
                X_train[col].dropna().to_numpy(),
                X_test[col].dropna().to_numpy(),
            )
            ks[col] = round(float(p), 6)
            if p < min_p:
                min_p, worst_ks_col = p, col
        evidence["ks_drift"] = ks
        checks.append(Check(
            name="ks_drift",
            status=threshold_status(min_p, ART10_KS_PVALUE_WARN, ART10_KS_PVALUE_FAIL, higher_is_worse=False),
            value=round(min_p, 6),
            threshold=ART10_KS_PVALUE_WARN,
            message=f"Min KS p-value {min_p:.4f} (col: {worst_ks_col})",
        ))

    return AuditFinding(article="10", checks=checks, evidence=evidence)
