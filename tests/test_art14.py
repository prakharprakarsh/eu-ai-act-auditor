from auditor.article_14 import audit
from auditor.base import AuditFinding


def test_returns_audit_finding(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert isinstance(finding, AuditFinding)


def test_article_number(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert finding.article == "14"


def test_brier_check_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "brier_score" in {c.name for c in finding.checks}


def test_ece_check_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "calibration_ece" in {c.name for c in finding.checks}


def test_brier_value_in_range(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    brier = next(c for c in finding.checks if c.name == "brier_score")
    assert 0.0 <= brier.value <= 1.0


def test_ece_value_in_range(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    ece = next(c for c in finding.checks if c.name == "calibration_ece")
    assert 0.0 <= ece.value <= 1.0


def test_low_conf_rate_in_evidence(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "low_conf_rate" in finding.evidence
    assert 0.0 <= finding.evidence["low_conf_rate"] <= 1.0


def test_abstention_rate_in_evidence(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "abstention_rate" in finding.evidence
    assert 0.0 <= finding.evidence["abstention_rate"] <= 1.0


def test_status_resolves(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    # status is set and comes from worst check
    from auditor.base import AuditStatus
    assert finding.status in {AuditStatus.PASS, AuditStatus.WARN, AuditStatus.FAIL}
