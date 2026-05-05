from __future__ import annotations

import datetime
import sys
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import auditor.constants as C
from auditor.base import AuditFinding, AuditStatus
from reports.charts import (
    plot_calibration_curve,
    plot_class_balance,
    plot_fairness_metrics,
    plot_protected_coverage,
    plot_shap_importance,
    plot_shap_local_samples,
)

_ACCENT = colors.HexColor("#1F4E79")
_GRAY = colors.HexColor("#555555")
_STATUS_COLOR = {
    AuditStatus.PASS: colors.HexColor("#2E7D32"),
    AuditStatus.WARN: colors.HexColor("#E65100"),
    AuditStatus.FAIL: colors.HexColor("#C62828"),
}
_ARTICLE_META = {
    "10": {
        "title": "Article 10 — Data and Data Governance",
        "desc": (
            "Article 10 requires that training, validation, and test datasets for high-risk AI "
            "systems are subject to data governance practices. Datasets must be relevant, "
            "representative, sufficiently free of errors, and complete. Known biases that may "
            "affect model outputs must be identified and mitigated where possible."
        ),
        "source": "Source: Regulation (EU) 2024/1689 — Article 10",
    },
    "13": {
        "title": "Article 13 — Transparency and Provision of Information",
        "desc": (
            "Article 13 requires that high-risk AI systems are sufficiently transparent to "
            "allow deployers to interpret and act on outputs. Providers must supply clear "
            "instructions for use and document capabilities, limitations, and performance "
            "characteristics, including with respect to protected groups."
        ),
        "source": "Source: Regulation (EU) 2024/1689 — Article 13",
    },
    "14": {
        "title": "Article 14 — Human Oversight",
        "desc": (
            "Article 14 requires that high-risk AI systems are designed so natural persons can "
            "effectively oversee their operation. Systems must allow reviewers to detect and "
            "correct errors, particularly for low-confidence predictions. Calibration quality "
            "enables operators to trust and act on model uncertainty signals."
        ),
        "source": "Source: Regulation (EU) 2024/1689 — Article 14",
    },
    "15": {
        "title": "Article 15 — Accuracy, Robustness and Cybersecurity",
        "desc": (
            "Article 15 requires high-risk AI systems to achieve appropriate accuracy, "
            "robustness, and security. Performance must remain stable under foreseeable "
            "input perturbations, and outcomes must be consistent across demographic groups. "
            "Providers must declare the metrics and reference datasets used for evaluation."
        ),
        "source": "Source: Regulation (EU) 2024/1689 — Article 15",
    },
}

# ---------------------------------------------------------------------------
# Module-level cell paragraph styles
# ---------------------------------------------------------------------------

_PC = ParagraphStyle("_PC", fontName="Helvetica", fontSize=8,
                     textColor=colors.HexColor("#333333"), leading=11)
_PH = ParagraphStyle("_PH", fontName="Helvetica-Bold", fontSize=8,
                     textColor=colors.white, leading=11)
_PS = ParagraphStyle("_PS", fontName="Helvetica-Bold", fontSize=8,
                     textColor=colors.white, leading=11, alignment=TA_CENTER)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_value(v) -> str:
    if isinstance(v, str):
        return v
    try:
        fv = float(v)
        if fv != 0.0 and abs(fv) < 0.001:
            return f"{fv:.2e}"
        return f"{fv:.3f}"
    except (TypeError, ValueError):
        return str(v)


def _format_limit(value, threshold) -> str:
    if threshold is None:
        return "—"
    try:
        fv, ft = float(value), float(threshold)
        comparator = "≥" if fv >= ft else "≤"
        return f"{comparator} {_format_value(ft)}"
    except (TypeError, ValueError):
        return str(threshold)


# ---------------------------------------------------------------------------
# Numbered canvas
# ---------------------------------------------------------------------------

class _AuditCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_decoration(n)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_decoration(self, page_count: int) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        footer = (
            f"Auditor v0.1.0 · Generated {ts}"
            f"   Page {self._pageNumber} of {page_count}"
        )
        w, h = A4
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawCentredString(w / 2, 1.0 * cm, footer)
        if self._pageNumber > 1:
            self.drawString(2 * cm, h - 1.3 * cm, "EU AI Act Compliance Audit Report")
            self.drawRightString(w - 2 * cm, h - 1.3 * cm,
                                 datetime.date.today().isoformat())
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(2 * cm, h - 1.5 * cm, w - 2 * cm, h - 1.5 * cm)
        self.restoreState()


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _make_styles() -> dict[str, ParagraphStyle]:
    def S(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, **kw)

    return {
        "Title": S("RPTitle", fontName="Helvetica-Bold", fontSize=22,
                   textColor=_ACCENT, alignment=TA_CENTER, spaceAfter=2),
        "Subtitle": S("RPSubtitle", fontName="Helvetica", fontSize=12,
                      textColor=_GRAY, alignment=TA_CENTER, spaceAfter=4),
        "H1": S("RPH1", fontName="Helvetica-Bold", fontSize=14,
                textColor=_ACCENT, spaceBefore=8, spaceAfter=4),
        "H2": S("RPH2", fontName="Helvetica-Bold", fontSize=11,
                textColor=colors.HexColor("#333333"), spaceBefore=6, spaceAfter=3),
        "Body": S("RPBody", fontName="Helvetica", fontSize=9,
                  textColor=colors.HexColor("#333333"), leading=13, spaceAfter=4),
        "Source": S("RPSource", fontName="Helvetica-Oblique", fontSize=8,
                    textColor=_GRAY, alignment=TA_LEFT),
        "Disclaimer": S("RPDisclaimer", fontName="Helvetica", fontSize=8,
                        textColor=colors.HexColor("#555555"), leading=12),
        "TOCEntry": S("RPTOCEntry", fontName="Helvetica", fontSize=10, leading=18),
        "TOCPage": S("RPTOCPage", fontName="Helvetica-Bold", fontSize=10,
                     leading=18, alignment=TA_RIGHT),
    }


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def _status_badge(status: AuditStatus, large: bool = False) -> Table:
    fs = 16 if large else 10
    pad = 10 if large else 5
    t = Table([[status.value]], colWidths=[5 * cm if large else 3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _STATUS_COLOR[status]),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
    ]))
    return t


def _check_table(checks: list) -> Table:
    # Column widths sum to 17 cm.
    col_w = [4.0 * cm, 1.8 * cm, 2.0 * cm, 2.5 * cm, 6.7 * cm]
    header = [
        Paragraph("Check", _PH),
        Paragraph("Status", _PS),
        Paragraph("Value", _PH),
        Paragraph("Limit", _PH),
        Paragraph("Message", _PH),
    ]
    rows = [header]
    for c in checks:
        rows.append([
            Paragraph(c.name.replace("_", " "), _PC),
            Paragraph(c.status.value, _PS),
            Paragraph(_format_value(c.value), _PC),
            Paragraph(_format_limit(c.value, c.threshold), _PC),
            Paragraph(c.message, _PC),
        ])
    t = Table(rows, colWidths=col_w, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
    ])
    for i, check in enumerate(checks):
        style.add("BACKGROUND", (1, i + 1), (1, i + 1), _STATUS_COLOR[check.status])
    t.setStyle(style)
    return t


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _cover(findings: list[AuditFinding], model_name: str,
           dataset_name: str, styles: dict) -> list:
    from auditor.base import _SEVERITY
    overall = (max(findings, key=lambda f: _SEVERITY[f.status]).status
               if findings else AuditStatus.PASS)
    story: list = []
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("EU AI Act Compliance Audit Report", styles["Title"]))
    # Explicit spacer prevents the HR from overlapping the title descender.
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=3, color=_ACCENT, spaceAfter=10))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"Model: <b>{model_name}</b>", styles["Subtitle"]))
    story.append(Paragraph(f"Dataset: <b>{dataset_name}</b>", styles["Subtitle"]))
    story.append(Paragraph(
        f"Audit date: {datetime.date.today().isoformat()}", styles["Subtitle"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Overall Audit Status", styles["H2"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(_status_badge(overall, large=True))
    story.append(Spacer(1, 2.0 * cm))

    disc_text = (
        "<b>Prototype Disclaimer.</b> This report is produced by an automated developer "
        "tool and does not constitute legal certification, a CE marking process, or a "
        "formal compliance assessment under Regulation (EU) 2024/1689 (EU AI Act). "
        "Thresholds and interpretations reflect engineering heuristics, not binding "
        "regulatory guidance. Seek qualified legal counsel for compliance determinations."
    )
    disc_table = Table(
        [[Paragraph(disc_text, styles["Disclaimer"])]],
        colWidths=[17 * cm],
    )
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E1")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#F9A825")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(disc_table)
    return story


def _toc(findings: list[AuditFinding], styles: dict) -> list:
    """Table of Contents page. Page numbers assume one page per section."""
    present = {f.article for f in findings}
    article_labels = {
        "10": "Article 10 — Data and Data Governance",
        "13": "Article 13 — Transparency and Provision of Information",
        "14": "Article 14 — Human Oversight",
        "15": "Article 15 — Accuracy, Robustness and Cybersecurity",
    }
    # Fixed offsets: Cover=1, TOC=2, ExecSummary=3, articles start at 4.
    entries: list[tuple[str, int]] = [
        ("Cover Page", 1),
        ("Executive Summary", 3),
    ]
    page = 4
    for art in ("10", "13", "14", "15"):
        if art in present:
            entries.append((article_labels[art], page))
            page += 1
    entries.append(("Appendix", page))

    story: list = []
    story.append(Paragraph("Contents", styles["H1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT, spaceAfter=10))
    story.append(Spacer(1, 0.3 * cm))

    rows = []
    for name, pg in entries:
        rows.append([
            Paragraph(name, styles["TOCEntry"]),
            Paragraph(str(pg), styles["TOCPage"]),
        ])
    t = Table(rows, colWidths=[14.5 * cm, 2.5 * cm])
    ts = TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#DDDDDD")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ])
    # Indent article entries slightly.
    for i, (name, _) in enumerate(entries):
        if name.startswith("Article"):
            ts.add("LEFTPADDING", (0, i), (0, i), 16)
    t.setStyle(ts)
    story.append(t)
    return story


def _executive_summary(findings: list[AuditFinding], styles: dict) -> list:
    story: list = []
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT, spaceAfter=6))

    all_checks = [c for f in findings for c in f.checks]
    n_warn = sum(1 for c in all_checks if c.status == AuditStatus.WARN)
    n_fail = sum(1 for c in all_checks if c.status == AuditStatus.FAIL)
    story.append(Paragraph(
        f"Total checks run: <b>{len(all_checks)}</b>   "
        f"Warnings: <b>{n_warn}</b>   Failures: <b>{n_fail}</b>",
        styles["Body"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    topic = {"10": "Data Governance", "13": "Transparency",
              "14": "Human Oversight", "15": "Accuracy & Robustness"}
    col_w = [1.5 * cm, 9.0 * cm, 4.0 * cm, 2.5 * cm]
    rows = [[
        Paragraph("Article", _PH),
        Paragraph("Requirement", _PH),
        Paragraph("Status", _PS),
        Paragraph("Checks", _PH),
    ]]
    for f in sorted(findings, key=lambda x: x.article):
        rows.append([
            Paragraph(f"Art. {f.article}", _PC),
            Paragraph(topic.get(f.article, ""), _PC),
            Paragraph(f.status.value, _PS),
            Paragraph(str(len(f.checks)), _PC),
        ])
    t = Table(rows, colWidths=col_w)
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ALIGN", (3, 0), (3, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])
    fmap = {f.article: f for f in findings}
    for i, art in enumerate(
        [f.article for f in sorted(findings, key=lambda x: x.article)], 1
    ):
        ts.add("BACKGROUND", (2, i), (2, i), _STATUS_COLOR[fmap[art].status])
    t.setStyle(ts)
    story.append(t)
    return story


def _article_charts(
    article: str, ev: dict, tmpdir: Path
) -> list[tuple[Path, float, float]]:
    """Return (path, width, height) tuples with article-appropriate sizing."""
    charts: list[tuple[Path, float, float]] = []
    try:
        if article == "10":
            if "class_distribution" in ev:
                p = plot_class_balance(ev["class_distribution"], tmpdir)
                charts.append((p, 7 * cm, 4.0 * cm))
            if ev.get("protected_coverage"):
                p = plot_protected_coverage(ev["protected_coverage"], tmpdir)
                charts.append((p, 14 * cm, 4.5 * cm))
        elif article == "13":
            if "shap_global_importance" in ev:
                top = list(ev["shap_global_importance"].items())
                p = plot_shap_importance(top, tmpdir)
                charts.append((p, 13 * cm, 5.0 * cm))
            if "shap_local" in ev:
                p = plot_shap_local_samples(ev["shap_local"], tmpdir)
                charts.append((p, 14 * cm, 5.0 * cm))
        elif article == "14":
            if "y_true" in ev and "y_prob" in ev:
                p = plot_calibration_curve(ev["y_true"], ev["y_prob"], tmpdir=tmpdir)
                charts.append((p, 11 * cm, 6.5 * cm))
        elif article == "15":
            if "demographic_parity" in ev and "equalized_odds" in ev:
                metrics = {
                    attr: {"dpd": dpd_val, "eod": ev["equalized_odds"].get(attr, 0)}
                    for attr, dpd_val in ev["demographic_parity"].items()
                }
                p = plot_fairness_metrics(metrics, tmpdir)
                charts.append((p, 12 * cm, 5.5 * cm))
    except Exception as exc:
        print(f"[PDF] chart failed for article {article}: {exc!r}", file=sys.stderr)
    return charts


def _article_section(finding: AuditFinding, styles: dict, tmpdir: Path) -> list:
    meta = _ARTICLE_META.get(finding.article, {
        "title": f"Article {finding.article}",
        "desc": "",
        "source": "",
    })
    story: list = []

    header_block = KeepTogether([
        Paragraph(meta["title"], styles["H1"]),
        HRFlowable(width="100%", thickness=0.5, color=_ACCENT, spaceAfter=4),
        Paragraph(meta["desc"], styles["Body"]),
        Spacer(1, 0.2 * cm),
        _status_badge(finding.status),
        Spacer(1, 0.3 * cm),
    ])
    story.append(header_block)

    if finding.checks:
        story.append(Paragraph("Audit Checks", styles["H2"]))
        story.append(_check_table(finding.checks))
        story.append(Spacer(1, 0.4 * cm))

    chart_items: list[Image] = []
    for path, w, h in _article_charts(finding.article, finding.evidence, tmpdir):
        if path and path.exists():
            chart_items.append(Image(str(path), width=w, height=h))

    source_para = Paragraph(meta["source"], styles["Source"])

    if chart_items:
        for img in chart_items[:-1]:
            story.append(img)
            story.append(Spacer(1, 0.3 * cm))
        story.append(KeepTogether([
            chart_items[-1],
            Spacer(1, 0.2 * cm),
            Spacer(1, 0.3 * cm),
            source_para,
        ]))
    else:
        story.append(Spacer(1, 0.3 * cm))
        story.append(source_para)

    return story


def _appendix(findings_by_art: dict, styles: dict) -> list:
    story: list = []
    story.append(Paragraph("Appendix", styles["H1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_ACCENT, spaceAfter=6))

    art13 = findings_by_art.get("13")
    if art13 and "model_card" in art13.evidence:
        story.append(Paragraph("A. Model Card", styles["H2"]))
        card = art13.evidence["model_card"]
        rows = []
        for key, val in card.items():
            val_str = (", ".join(str(v) for v in val)
                       if isinstance(val, list) else str(val))
            rows.append([
                Paragraph(key.replace("_", " ").title(), _PC),
                Paragraph(val_str, _PC),
            ])
        t = Table(rows, colWidths=[5 * cm, 12 * cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#F5F7FA")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("B. Audit Configuration (Thresholds)", styles["H2"]))
    trows = [
        [Paragraph("Parameter", _PH), Paragraph("Threshold", _PH), Paragraph("Article", _PH)],
        ["Class balance warn", f"< {C.ART10_CLASS_BALANCE_WARN:.0%}", "10"],
        ["Class balance fail", f"< {C.ART10_CLASS_BALANCE_FAIL:.0%}", "10"],
        ["Protected coverage warn", f"< {C.ART10_COVERAGE_WARN:.0%}", "10"],
        ["Missing value rate warn", f"> {C.ART10_MISSING_WARN:.0%}", "10"],
        ["Brier score warn / fail", f"> {C.ART14_BRIER_WARN} / > {C.ART14_BRIER_FAIL}", "14"],
        ["ECE warn / fail", f"> {C.ART14_ECE_WARN} / > {C.ART14_ECE_FAIL}", "14"],
        ["Accuracy warn / fail",
         f"< {C.ART15_ACCURACY_WARN:.0%} / < {C.ART15_ACCURACY_FAIL:.0%}", "15"],
        ["Fairness metric warn / fail",
         f"> {C.ART15_FAIRNESS_WARN} / > {C.ART15_FAIRNESS_FAIL}", "15"],
        ["Robustness stability warn", f"< {C.ART15_ROBUSTNESS_WARN:.0%}", "15"],
        ["Perturbation noise σ", str(C.ART15_NOISE_SIGMA), "15"],
    ]
    t2 = Table(trows, colWidths=[8 * cm, 5 * cm, 4 * cm], repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    return story


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    findings: list[AuditFinding],
    model_name: str,
    dataset_name: str,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    styles = _make_styles()
    findings_by_art = {f.article: f for f in findings}

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="EU AI Act Compliance Audit Report",
        author="EU AI Act Auditor (prototype)",
    )

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        story: list = []
        story.extend(_cover(findings, model_name, dataset_name, styles))
        story.append(PageBreak())
        story.extend(_toc(findings, styles))
        story.append(PageBreak())
        story.extend(_executive_summary(findings, styles))
        story.append(PageBreak())
        for article in ("10", "13", "14", "15"):
            finding = findings_by_art.get(article)
            if finding is None:
                continue
            story.extend(_article_section(finding, styles, tmpdir))
            story.append(PageBreak())
        story.extend(_appendix(findings_by_art, styles))
        doc.build(story, canvasmaker=_AuditCanvas)

    return output_path
