import pytest
from auditor.base import AuditFinding, AuditStatus, Check


def make_check(status: AuditStatus) -> Check:
    return Check(name="x", status=status, value=0.5, threshold=0.8, message="msg")


# --- Check construction ---

def test_check_stores_all_fields():
    c = Check(name="class_balance", status=AuditStatus.WARN, value=0.3, threshold=0.4, message="imbalanced")
    assert c.name == "class_balance"
    assert c.status == AuditStatus.WARN
    assert c.value == 0.3
    assert c.threshold == 0.4
    assert c.message == "imbalanced"


def test_check_threshold_can_be_none():
    c = Check(name="drift", status=AuditStatus.PASS, value="n/a", threshold=None, message="ok")
    assert c.threshold is None


# --- AuditFinding construction ---

def test_finding_stores_article_and_evidence():
    finding = AuditFinding(article="10", evidence={"n_samples": 500})
    assert finding.article == "10"
    assert finding.evidence["n_samples"] == 500


def test_finding_checks_default_to_empty_list():
    finding = AuditFinding(article="13")
    assert finding.checks == []


def test_finding_evidence_default_to_empty_dict():
    finding = AuditFinding(article="14")
    assert finding.evidence == {}


# --- AuditFinding.status worst-case logic ---

def test_status_is_pass_when_no_checks():
    finding = AuditFinding(article="10")
    assert finding.status == AuditStatus.PASS


def test_status_is_pass_when_all_pass():
    finding = AuditFinding(article="10", checks=[make_check(AuditStatus.PASS)] * 3)
    assert finding.status == AuditStatus.PASS


def test_status_is_warn_when_worst_is_warn():
    finding = AuditFinding(
        article="10",
        checks=[make_check(AuditStatus.PASS), make_check(AuditStatus.WARN)],
    )
    assert finding.status == AuditStatus.WARN


def test_status_is_fail_when_any_check_fails():
    finding = AuditFinding(
        article="15",
        checks=[
            make_check(AuditStatus.PASS),
            make_check(AuditStatus.WARN),
            make_check(AuditStatus.FAIL),
        ],
    )
    assert finding.status == AuditStatus.FAIL


def test_status_is_fail_with_single_fail_check():
    finding = AuditFinding(article="14", checks=[make_check(AuditStatus.FAIL)])
    assert finding.status == AuditStatus.FAIL


def test_status_fail_dominates_warn():
    finding = AuditFinding(
        article="15",
        checks=[make_check(AuditStatus.WARN), make_check(AuditStatus.FAIL)],
    )
    assert finding.status == AuditStatus.FAIL


# --- AuditStatus enum ---

def test_audit_status_str_values():
    assert AuditStatus.PASS == "PASS"
    assert AuditStatus.WARN == "WARN"
    assert AuditStatus.FAIL == "FAIL"


def test_audit_status_ordering():
    from auditor.base import _SEVERITY
    assert _SEVERITY[AuditStatus.PASS] < _SEVERITY[AuditStatus.WARN] < _SEVERITY[AuditStatus.FAIL]
