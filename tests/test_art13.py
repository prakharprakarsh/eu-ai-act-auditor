from auditor.article_13 import audit
from auditor.base import AuditFinding, AuditStatus


def test_returns_audit_finding(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert isinstance(finding, AuditFinding)


def test_article_number(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert finding.article == "13"


def test_shap_check_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "shap_global_importance" in {c.name for c in finding.checks}


def test_model_card_check_present(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "model_card_completeness" in {c.name for c in finding.checks}


def test_shap_runs_successfully(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    shap_check = next(c for c in finding.checks if c.name == "shap_global_importance")
    assert shap_check.status == AuditStatus.PASS


def test_shap_importance_top5_in_evidence(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "shap_global_importance" in finding.evidence
    assert len(finding.evidence["shap_global_importance"]) == 5


def test_shap_local_in_evidence(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    assert "shap_local" in finding.evidence
    local = finding.evidence["shap_local"]
    assert "sample_0" in local
    # Each local explanation has one value per feature
    X_cols = set(X.columns)
    assert set(local["sample_0"].keys()) == X_cols


def test_model_card_in_evidence(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    card = finding.evidence["model_card"]
    assert "model_type" in card
    assert "feature_list" in card
    assert "protected_attrs_documented" in card
    assert set(protected_attrs.keys()).issubset(set(card["protected_attrs_documented"]))


def test_model_card_completeness_passes(trained_hiring_model, hiring_data, protected_attrs):
    X, y = hiring_data
    finding = audit(trained_hiring_model, X, y, protected_attrs)
    card_check = next(c for c in finding.checks if c.name == "model_card_completeness")
    assert card_check.status == AuditStatus.PASS
