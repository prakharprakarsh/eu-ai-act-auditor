from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from auditor.article_10 import audit as audit_10
from auditor.article_13 import audit as audit_13
from auditor.article_14 import audit as audit_14
from auditor.article_15 import audit as audit_15
from auditor.base import AuditFinding, AuditStatus
from reports.charts import plot_fairness_metrics, plot_shap_importance
from reports.pdf_generator import generate_report

_DATA = Path(__file__).parent.parent / "data" / "processed"
_PROTECTED = ["gender", "age_band", "ethnicity_proxy"]
_SEVERITY = {"PASS": 0, "WARN": 1, "FAIL": 2}
_TOPIC = {"10": "Data Governance", "13": "Transparency",
          "14": "Human Oversight", "15": "Accuracy & Robustness"}
_BADGE_COLOR = {
    AuditStatus.PASS: "#2E7D32",
    AuditStatus.WARN: "#E65100",
    AuditStatus.FAIL: "#C62828",
}


def _overall(findings: list[AuditFinding]) -> AuditStatus:
    return max(findings, key=lambda f: _SEVERITY[f.status.value]).status


def _status_badge(status: AuditStatus) -> str:
    bg = _BADGE_COLOR[status]
    return (
        f'<div style="background:{bg};color:white;font-size:2rem;font-weight:bold;'
        f'text-align:center;padding:1rem 2.5rem;border-radius:10px;'
        f'display:inline-block;letter-spacing:0.05em;margin:0.5rem 0">'
        f"{status.value}</div>"
    )


def _summary_df(findings: list[AuditFinding]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Article": f"Art. {f.article}", "Requirement": _TOPIC.get(f.article, ""),
         "Status": f.status.value, "Checks": len(f.checks)}
        for f in sorted(findings, key=lambda x: x.article)
    ])


def _make_charts(findings: list[AuditFinding], tmpdir: Path) -> tuple[str | None, str | None]:
    fmap = {f.article: f for f in findings}
    shap_path = fairness_path = None
    art13 = fmap.get("13")
    if art13 and "shap_global_importance" in art13.evidence:
        shap_path = str(plot_shap_importance(
            list(art13.evidence["shap_global_importance"].items()), tmpdir
        ))
    art15 = fmap.get("15")
    if art15 and "demographic_parity" in art15.evidence:
        metrics = {
            attr: {"dpd": dpd, "eod": art15.evidence["equalized_odds"].get(attr, 0)}
            for attr, dpd in art15.evidence["demographic_parity"].items()
        }
        fairness_path = str(plot_fairness_metrics(metrics, tmpdir))
    return shap_path, fairness_path


def _run_pipeline(model, X_test, y_test, X_train, protected_attrs, progress):
    tmpdir = Path(tempfile.mkdtemp())
    progress(0.10, desc="Auditing Article 10... (1/4)")
    f10 = audit_10(model, X_test, y_test, protected_attrs, X_train)
    progress(0.35, desc="Auditing Article 13... (2/4)")
    f13 = audit_13(model, X_test, y_test, protected_attrs)
    progress(0.60, desc="Auditing Article 14... (3/4)")
    f14 = audit_14(model, X_test, y_test, protected_attrs)
    progress(0.80, desc="Auditing Article 15... (4/4)")
    f15 = audit_15(model, X_test, y_test, protected_attrs)
    findings = [f10, f13, f14, f15]
    progress(0.92, desc="Generating PDF report...")
    pdf = str(generate_report(
        findings,
        model_name=type(model).__name__,
        dataset_name="CV Screening Dataset (synthetic)",
        output_path=tmpdir / "audit_report.pdf",
    ))
    shap, fair = _make_charts(findings, tmpdir)
    return findings, pdf, shap, fair


def run_bundled(progress=gr.Progress()):
    try:
        with open(_DATA / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        X_test = pd.read_parquet(_DATA / "X_test.parquet")
        y_test = pd.read_parquet(_DATA / "y_test.parquet").squeeze()
        X_train = pd.read_parquet(_DATA / "X_train.parquet")
        attrs = {c: X_test[c] for c in _PROTECTED if c in X_test.columns}
        findings, pdf, shap, fair = _run_pipeline(
            model, X_test, y_test, X_train, attrs, progress
        )
        progress(1.0, desc="Done")
        return (
            gr.update(visible=True),
            _status_badge(_overall(findings)), _summary_df(findings), shap, fair, pdf, "",
        )
    except Exception as exc:
        return gr.update(visible=False), None, None, None, None, None, f"**Error:** {exc}"


def run_custom(model_f, xtest_f, ytest_f, xtrain_f, selected, progress=gr.Progress()):
    try:
        if not model_f or not xtest_f or not ytest_f:
            raise ValueError("model.pkl, X_test.parquet, and y_test.parquet are required.")
        with open(model_f, "rb") as fh:
            model = pickle.load(fh)
        if not (hasattr(model, "predict") and hasattr(model, "predict_proba")):
            raise ValueError("Model must expose predict() and predict_proba() (sklearn API).")
        X_test = pd.read_parquet(xtest_f)
        y_test = pd.read_parquet(ytest_f).squeeze()
        if len(X_test) != len(y_test):
            raise ValueError(f"Row count mismatch: X_test={len(X_test)}, y_test={len(y_test)}.")
        X_train = pd.read_parquet(xtrain_f) if xtrain_f else None
        if not selected:
            raise ValueError("Select at least one protected attribute column.")
        bad = [c for c in selected if c not in X_test.columns]
        if bad:
            raise ValueError(f"Columns not found in X_test: {bad}")
        attrs = {c: X_test[c] for c in selected}
        findings, pdf, shap, fair = _run_pipeline(
            model, X_test, y_test, X_train, attrs, progress
        )
        progress(1.0, desc="Done")
        return (
            gr.update(visible=True),
            _status_badge(_overall(findings)), _summary_df(findings), shap, fair, pdf, "",
        )
    except Exception as exc:
        return gr.update(visible=False), None, None, None, None, None, f"**Error:** {exc}"


def update_protected_choices(f):
    if not f:
        return gr.update(choices=[])
    try:
        return gr.update(choices=list(pd.read_parquet(f).columns))
    except Exception:
        return gr.update(choices=[])
