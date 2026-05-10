from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from app_helpers import (  # noqa: E402
    ensure_bundled_artifacts,
    run_bundled,
    run_custom,
    update_protected_choices,
)

ensure_bundled_artifacts()

# ── Tab content ───────────────────────────────────────────────────────────────

_INTRO = """\
This is a demonstration audit on a **synthetic CV-screening model** with deliberately injected \
gender and age bias, designed to show the auditor catching real Annex III §4 high-risk violations. \
The dataset and model are purely synthetic — no real personal data is involved.
"""

_ABOUT = """
## What is the EU AI Act?

Regulation (EU) 2024/1689 (the EU AI Act) is the world's first comprehensive legal framework for
artificial intelligence, adopted on 13 June 2024 and published in the Official Journal on 12 July
2024. It establishes a risk-based classification system and imposes binding obligations on providers
and deployers of AI systems placed on the EU market or affecting persons in the EU. Key enforcement
milestones: prohibitions on unacceptable-risk AI applied from **6 February 2025**; General-Purpose
AI (GPAI) model rules apply from **2 August 2026**; high-risk system requirements under Annex III
apply from **2 August 2027**.

## What Does "High-Risk" Mean?

Annex III designates eight categories of high-risk AI systems subject to the full obligations of
Chapter III (Articles 6–27). The categories are:

1. Biometric identification and categorisation of natural persons
2. Management and operation of critical infrastructure
3. Education and vocational training
4. **Employment, workers management, and access to self-employment (§4)**
5. Access to essential private services — including credit scoring and insurance risk assessment
6. Law enforcement
7. Migration, asylum, and border control management
8. Administration of justice and democratic processes

This auditor's reference implementation targets **Annex III §4(a)**: CV/résumé screening and
automated candidate ranking systems, which are explicitly named as high-risk.

## What This Auditor Covers

| Article | Title | What is checked |
|---------|-------|-----------------|
| **Art. 10** | Data and Data Governance | Class balance, protected-group coverage, missing-value rates, KS drift (train → test) |
| **Art. 13** | Transparency and Provision of Information | SHAP global importance, top-5 feature explanations, auto-generated model card |
| **Art. 14** | Human Oversight | Brier score, Expected Calibration Error (ECE), low-confidence flag rate |
| **Art. 15** | Accuracy, Robustness and Cybersecurity | Accuracy, F1, demographic parity difference, equalized odds difference, Gaussian perturbation stability |

## What This Auditor Does NOT Cover

- **Article 9** — Risk management system: requires a documented, ongoing organisational process;
  not automatable from model artefacts alone.
- **Articles 11–12** — Technical documentation and record-keeping: the generated model card
  partially addresses Art. 11, but full compliance requires provider-level documentation packages.
- **Article 16+** — Provider and deployer obligations: registration in the EU database, conformity
  assessments, CE marking, and post-market monitoring are organisational and legal processes
  outside the scope of a technical auditing tool.

## Limitations

This auditor is a **developer prototype**, not a legal instrument:

- Thresholds (e.g. demographic parity difference ≥ 0.10 → WARN) are engineering heuristics
  informed by fairness literature, not binding regulatory guidance from a supervisory authority.
- A **PASS** result does not constitute compliance under Regulation (EU) 2024/1689 and cannot
  substitute for a formal conformity assessment or CE-marking process.
- The tool has not been audited or endorsed by any notified body, national supervisory authority,
  or legal counsel. It is intended for ML practitioners exploring the technical implications of
  the Act, not for compliance officers making legal determinations.
- **Seek qualified legal counsel** before making any compliance representations to regulators,
  clients, or business partners.

## Sources and Links

- [Regulation (EU) 2024/1689 — Full text (EUR-Lex)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- [GitHub repo] *(placeholder — link to be added on publication)*
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    title="EU AI Act Compliance Auditor",
) as demo:
    gr.Markdown("# EU AI Act Compliance Auditor")
    gr.Markdown(
        "*Prototype auditor for high-risk AI systems under Annex III — "
        "Regulation (EU) 2024/1689*"
    )
    gr.HTML("<hr style='border:none;border-top:2px solid #1F4E79;margin:0 0 8px 0;'>")

    with gr.Tabs():
        # ── Tab 1: Live Demo ──────────────────────────────────────────────────
        with gr.Tab("Live Demo"):
            gr.Markdown(_INTRO)
            run_btn = gr.Button(
                "Run audit on bundled hiring model", variant="primary", size="lg"
            )
            t1_error = gr.Markdown()
            with gr.Column(visible=False) as results_col:
                t1_badge = gr.HTML(label="Overall Status")
                t1_table = gr.Dataframe(
                    label="Executive Summary",
                    headers=["Article", "Requirement", "Status", "Checks"],
                    col_count=(4, "fixed"),
                    datatype=["str", "str", "str", "number"],
                    interactive=False,
                )
                with gr.Row():
                    t1_shap = gr.Image(label="SHAP Global Importance")
                    t1_fair = gr.Image(label="Fairness Metrics (Art. 15)")
                t1_pdf = gr.File(label="📄 Download Full Report (PDF)")

        # ── Tab 2: Custom Model ───────────────────────────────────────────────
        with gr.Tab("Run on Your Own Model"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_model = gr.File(
                        label="model.pkl (sklearn-compatible)", file_types=[".pkl"]
                    )
                    t2_xtest = gr.File(
                        label="X_test.parquet", file_types=[".parquet"]
                    )
                    t2_ytest = gr.File(
                        label="y_test.parquet", file_types=[".parquet"]
                    )
                    t2_xtrain = gr.File(
                        label="X_train.parquet (optional — enables KS drift check)",
                        file_types=[".parquet"],
                    )
                    t2_protected = gr.CheckboxGroup(
                        choices=[], label="Select protected attribute columns"
                    )
                    run_custom_btn = gr.Button("Run Audit", variant="primary")
            t2_error = gr.Markdown()
            with gr.Column(visible=False) as tab2_results_col:
                t2_badge = gr.HTML(label="Overall Status")
                t2_table = gr.Dataframe(
                    label="Executive Summary",
                    headers=["Article", "Requirement", "Status", "Checks"],
                    col_count=(4, "fixed"),
                    datatype=["str", "str", "str", "number"],
                    interactive=False,
                )
                with gr.Row():
                    t2_shap = gr.Image(label="SHAP Global Importance")
                    t2_fair = gr.Image(label="Fairness Metrics (Art. 15)")
                t2_pdf = gr.File(label="📄 Download Full Report (PDF)")

        # ── Tab 3: About ──────────────────────────────────────────────────────
        with gr.Tab("About the EU AI Act"):
            gr.Markdown(_ABOUT)

    # ── Event wiring ──────────────────────────────────────────────────────────
    _t1_out = [results_col, t1_badge, t1_table, t1_shap, t1_fair, t1_pdf, t1_error]
    run_btn.click(fn=run_bundled, inputs=[], outputs=_t1_out)

    t2_xtest.change(
        fn=update_protected_choices, inputs=[t2_xtest], outputs=[t2_protected]
    )
    _t2_out = [tab2_results_col, t2_badge, t2_table, t2_shap, t2_fair, t2_pdf, t2_error]
    run_custom_btn.click(
        fn=run_custom,
        inputs=[t2_model, t2_xtest, t2_ytest, t2_xtrain, t2_protected],
        outputs=_t2_out,
    )

demo.launch(share=False)
