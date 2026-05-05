"""
Tests for src/reports/pdf_generator.py.

All findings are constructed from mock data — no ML model or auditor module
is invoked, keeping the test suite fast.
"""
from __future__ import annotations

import pytest

from auditor.base import AuditFinding, AuditStatus, Check


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _check(name: str, status: AuditStatus, value=0.5, threshold=0.8, msg="ok") -> Check:
    return Check(name=name, status=status, value=value, threshold=threshold, message=msg)


@pytest.fixture(scope="module")
def mock_findings() -> list[AuditFinding]:
    return [
        AuditFinding(
            article="10",
            checks=[
                _check("class_balance", AuditStatus.PASS, 0.45, 0.30),
                _check("protected_coverage_gender", AuditStatus.PASS, 0.50, 0.05),
                _check("missing_values", AuditStatus.PASS, 0.0, 0.05),
            ],
            evidence={
                "class_distribution": {0: 0.55, 1: 0.45},
                "protected_coverage": {
                    "gender": {0: 0.50, 1: 0.50},
                    "age_band": {0: 0.35, 1: 0.45, 2: 0.20},
                },
                "missing_rates": {},
            },
        ),
        AuditFinding(
            article="13",
            checks=[
                _check("shap_global_importance", AuditStatus.PASS,
                       "top: prior_role_match", None, "SHAP computed."),
                _check("model_card_completeness", AuditStatus.PASS,
                       "7/7", "all fields non-empty", "Model card complete."),
            ],
            evidence={
                "shap_global_importance": {
                    "prior_role_match": 0.45,
                    "skills_overlap": 0.38,
                    "years_experience": 0.22,
                    "education_level": 0.18,
                    "gap_months": 0.12,
                },
                "model_card": {
                    "model_type": "XGBClassifier",
                    "training_date": "2026-05-05",
                    "feature_list": ["years_experience", "education_level",
                                     "prior_role_match", "skills_overlap",
                                     "gap_months", "gender", "age_band",
                                     "ethnicity_proxy"],
                    "target": "hired",
                    "intended_use": "CV/resume screening assistance (Annex III §4).",
                    "limitations": "May encode historical hiring biases.",
                    "protected_attrs_documented": ["gender", "age_band", "ethnicity_proxy"],
                },
            },
        ),
        AuditFinding(
            article="14",
            checks=[
                _check("brier_score", AuditStatus.PASS, 0.15, 0.20, "Brier score: 0.15"),
                _check("calibration_ece", AuditStatus.WARN, 0.12, 0.10, "ECE: 0.12"),
            ],
            evidence={
                "brier_score": 0.15,
                "ece": 0.12,
                "low_conf_rate": 0.08,
                "low_conf_threshold": 0.60,
                "abstention_rate": 0.03,
                "abstention_threshold": 0.70,
                # Include raw arrays so the calibration curve chart is exercised.
                "y_true": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                "y_prob": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.85, 0.15, 0.75],
            },
        ),
        AuditFinding(
            article="15",
            checks=[
                _check("accuracy", AuditStatus.PASS, 0.82, 0.75, "Accuracy: 0.82"),
                _check("demographic_parity_gender", AuditStatus.WARN,
                       0.14, 0.10, "DPD gender: 0.14"),
                _check("equalized_odds_gender", AuditStatus.WARN,
                       0.11, 0.10, "EOD gender: 0.11"),
                _check("robustness", AuditStatus.PASS, 0.92, 0.85, "Stability: 0.92"),
            ],
            evidence={
                "accuracy": 0.82,
                "precision": 0.79,
                "recall": 0.75,
                "f1": 0.77,
                "demographic_parity": {"gender": 0.14, "age_band": 0.07},
                "equalized_odds": {"gender": 0.11, "age_band": 0.05},
                "robustness_stability": 0.92,
                "noise_sigma": 0.05,
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generates_file(mock_findings, tmp_path):
    from reports.pdf_generator import generate_report
    out = tmp_path / "report.pdf"
    generate_report(mock_findings, "XGBClassifier", "SyntheticHiringData", out)
    assert out.exists()


def test_returns_output_path(mock_findings, tmp_path):
    from reports.pdf_generator import generate_report
    out = tmp_path / "report.pdf"
    result = generate_report(mock_findings, "XGBClassifier", "SyntheticHiringData", out)
    assert result == out


def test_file_is_valid_pdf(mock_findings, tmp_path):
    from reports.pdf_generator import generate_report
    out = tmp_path / "report.pdf"
    generate_report(mock_findings, "XGBClassifier", "SyntheticHiringData", out)
    with open(out, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", f"File does not start with %PDF-, got {header!r}"


def test_file_size_reasonable(mock_findings, tmp_path):
    from reports.pdf_generator import generate_report
    out = tmp_path / "report.pdf"
    generate_report(mock_findings, "XGBClassifier", "SyntheticHiringData", out)
    size = out.stat().st_size
    assert size > 10_000, f"PDF suspiciously small: {size:,} bytes"
    assert size < 5_000_000, f"PDF suspiciously large: {size:,} bytes"


def test_works_with_warn_and_fail_statuses(tmp_path):
    from reports.pdf_generator import generate_report
    findings = [
        AuditFinding(
            article="10",
            checks=[
                _check("class_balance", AuditStatus.WARN, 0.22, 0.30),
                _check("missing_values", AuditStatus.FAIL, 0.20, 0.15),
            ],
            evidence={},
        ),
        AuditFinding(
            article="15",
            checks=[_check("accuracy", AuditStatus.FAIL, 0.60, 0.65)],
            evidence={},
        ),
    ]
    out = tmp_path / "fail_report.pdf"
    generate_report(findings, "BadModel", "SmallDataset", out)
    assert out.exists()
    with open(out, "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_works_with_empty_findings(tmp_path):
    from reports.pdf_generator import generate_report
    out = tmp_path / "empty_report.pdf"
    generate_report([], "NoModel", "NoData", out)
    assert out.exists()
    with open(out, "rb") as f:
        assert f.read(5) == b"%PDF-"


def test_works_with_missing_evidence(tmp_path):
    # Findings with no evidence should not raise — charts are simply skipped.
    from reports.pdf_generator import generate_report
    findings = [
        AuditFinding(
            article=art,
            checks=[_check("test", AuditStatus.PASS)],
            evidence={},
        )
        for art in ("10", "13", "14", "15")
    ]
    out = tmp_path / "no_evidence.pdf"
    generate_report(findings, "Model", "Dataset", out)
    assert out.exists()
