from auditor.article_10 import audit
from auditor.base import AuditFinding, AuditStatus


def test_returns_audit_finding(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert isinstance(finding, AuditFinding)


def test_article_number(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert finding.article == "10"


def test_required_checks_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    names = {c.name for c in finding.checks}
    assert "class_balance" in names
    assert "missing_values" in names
    for attr in protected_attrs:
        assert f"protected_coverage_{attr}" in names


def test_missing_values_pass_on_clean_data(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    mv = next(c for c in finding.checks if c.name == "missing_values")
    assert mv.status == AuditStatus.PASS
    assert mv.value == 0.0


def test_no_ks_check_without_train(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "ks_drift" not in {c.name for c in finding.checks}


def test_ks_check_present_with_train(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs, X_train=X)
    assert "ks_drift" in {c.name for c in finding.checks}


def test_evidence_has_class_distribution(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "class_distribution" in finding.evidence
    assert "protected_coverage" in finding.evidence


def test_ks_same_dist_passes(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs, X_train=X)
    ks = next(c for c in finding.checks if c.name == "ks_drift")
    # Train == test, so KS p-values should be high → PASS
    assert ks.status == AuditStatus.PASS
