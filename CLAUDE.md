# EU AI Act Compliance Auditor

## Project Goal
Open-source compliance auditor for ML systems classified as **high-risk under Annex III of the EU AI Act**. Reference implementation: hiring/CV screening (Annex III §4 — employment, workers management, access to self-employment).

The auditor takes a trained model + test dataset and produces a structured PDF compliance report covering Articles 10, 13, 14, and 15.

This is a **prototype auditor**, not a legal substitute. The goal is to demonstrate deep, applied understanding of the EU AI Act for ML practitioners — not to provide legal certification.

## EU AI Act Articles in Scope

| Article | Title | What we audit |
|---------|-------|---------------|
| Art. 10 | Data and data governance | Class balance, protected attribute coverage, missing values, train/test drift |
| Art. 13 | Transparency and provision of information | SHAP global importance, top-K feature explanations, auto-generated model card |
| Art. 14 | Human oversight | Confidence calibration (Brier, ECE), low-confidence flagging, abstention rate |
| Art. 15 | Accuracy, robustness and cybersecurity | Performance metrics, demographic parity, equal opportunity, equalized odds, perturbation robustness |

## Tech Stack
- Python 3.12
- scikit-learn, XGBoost (model)
- SHAP (explainability)
- Fairlearn (fairness metrics)
- Evidently or scipy.stats.ks_2samp (drift)
- ReportLab (PDF generation)
- matplotlib (charts embedded in PDF)
- Gradio (UI)
- pytest (tests)

## Module Structure

```
eu-ai-act-auditor/
├── app.py                          # Gradio UI entry point
├── requirements.txt
├── src/
│   ├── auditor/
│   │   ├── __init__.py
│   │   ├── base.py                 # AuditFinding, Check, AuditStatus dataclasses
│   │   ├── constants.py            # Thresholds shared across all article modules
│   │   ├── art10_data.py           # Art. 10: data governance checks
│   │   ├── art13_transparency.py   # Art. 13: SHAP explainability + model card
│   │   ├── art14_oversight.py      # Art. 14: calibration + abstention
│   │   └── art15_robustness.py     # Art. 15: fairness metrics + perturbation
│   ├── models/
│   │   ├── __init__.py
│   │   └── hiring_model.py         # Reference hiring/CV screening model
│   └── reports/
│       ├── __init__.py
│       └── pdf_report.py           # ReportLab PDF assembly
├── tests/
│   ├── test_art10.py
│   ├── test_art13.py
│   ├── test_art14.py
│   └── test_art15.py
├── data/                           # Sample hiring dataset (synthetic)
├── notebooks/                      # Exploration / demo notebooks
└── docs/                           # Article reference notes
```

## Dev Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Gradio UI
python app.py

# Run all tests
pytest tests/

# Run a single audit module (for development)
python -m src.auditor.art10_data
```

## Key Design Decisions
- Each article maps to one module in `src/auditor/`. Each module exposes an `audit(model, X_test, y_test, protected_attrs) -> AuditFinding` function.
- `pdf_report.py` consumes the `AuditFinding` objects from all four modules and assembles the final PDF.
- The reference dataset is synthetic hiring data with `gender` and `age_group` as sensitive attributes.
- Pass/warn/fail thresholds are constants in `src/auditor/constants.py` — one place to adjust them all.

## Auditor Contract
Every article module exposes a single function:

```python
def audit(model, X_test, y_test, protected_attrs: dict) -> AuditFinding
```

`AuditFinding` is a dataclass with:
- `article: str` — e.g. "10"
- `checks: list[Check]` — each check has name, status (PASS/WARN/FAIL), value, threshold, message
- `status: AuditStatus` — worst status across checks
- `evidence: dict` — raw numbers, arrays, chart paths for the PDF

## Coding Conventions
- Type hints on every public function
- Dataclasses over dicts for structured returns
- One responsibility per module — no cross-article logic
- Functions over classes unless state is genuinely needed
- Tests use pytest fixtures from `tests/conftest.py` for shared model/data
- Keep each auditor module under 250 lines — if it grows beyond that, split it
- No hardcoded paths — use `pathlib.Path` and `constants.py` if needed
- **Never name a file `types.py`** — it shadows the Python standard library `types` module and causes subtle import bugs. Use `base.py` for shared dataclasses.

## Extending to Other Annex III Domains
A new contributor adding e.g. credit scoring should:
1. Add a new model under `src/models/` following the hiring_classifier pattern
2. Provide synthetic or open data with documented protected attributes
3. The four auditor modules should work unchanged — they take any sklearn-compatible model
4. Update `app.py` to add the new domain as a Gradio tab

## Out of Scope (Prototype Limits)
- Article 9 (risk management system) — organizational, not technical
- Article 11–12 (technical documentation, record-keeping) — partially covered by model card
- Article 16+ (provider obligations) — organizational
- Real legal certification — this is a developer tool, not a CE-marking process
