from auditor.article_15 import audit
from auditor.base import AuditFinding, AuditStatus


def test_returns_audit_finding(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert isinstance(finding, AuditFinding)


def test_article_number(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert finding.article == "15"


def test_accuracy_check_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "accuracy" in {c.name for c in finding.checks}


def test_demographic_parity_checks_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    names = {c.name for c in finding.checks}
    for attr in protected_attrs:
        assert f"demographic_parity_{attr}" in names


def test_equalized_odds_checks_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    names = {c.name for c in finding.checks}
    for attr in protected_attrs:
        assert f"equalized_odds_{attr}" in names


def test_robustness_check_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "robustness" in {c.name for c in finding.checks}


def test_performance_metrics_in_evidence(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert metric in finding.evidence


def test_bias_flagged_for_gender():
    # Use the full-size dataset so the model captures the bias signal above the 0.10 threshold.
    from models.synthetic_data import generate_hiring_dataset
    from models.hiring_classifier import train_hiring_model

    X, y = generate_hiring_dataset(n=5000, seed=42)
    model = train_hiring_model(X, y)
    attrs = {
        "gender": X["gender"],
        "age_band": X["age_band"],
        "ethnicity_proxy": X["ethnicity_proxy"],
    }
    finding = audit(model, X, y, attrs)
    dp = next(c for c in finding.checks if c.name == "demographic_parity_gender")
    assert dp.status in {AuditStatus.WARN, AuditStatus.FAIL}, (
        f"Expected WARN/FAIL for gender DPD, got {dp.status} (value={dp.value})"
    )


def test_overall_status_not_pass(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    # With injected bias the overall finding must be at least WARN
    assert finding.status in {AuditStatus.WARN, AuditStatus.FAIL}, (
        f"Expected WARN/FAIL overall, got {finding.status}"
    )


def test_robustness_stability_in_range(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    rob = next(c for c in finding.checks if c.name == "robustness")
    assert 0.0 <= rob.value <= 1.0
