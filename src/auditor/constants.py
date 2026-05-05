# Article 10 — Data Governance
ART10_CLASS_BALANCE_WARN = 0.30    # warn if minority class fraction < 30 %
ART10_CLASS_BALANCE_FAIL = 0.15    # fail if minority class fraction < 15 %
ART10_COVERAGE_WARN = 0.05         # warn if any protected group < 5 % of data
ART10_COVERAGE_FAIL = 0.02         # fail if any protected group < 2 % of data
ART10_MISSING_WARN = 0.05          # warn if any feature missing rate > 5 %
ART10_MISSING_FAIL = 0.15          # fail if any feature missing rate > 15 %
ART10_KS_PVALUE_WARN = 0.05        # warn if KS p-value < 0.05 (distribution shift)
ART10_KS_PVALUE_FAIL = 0.01        # fail if KS p-value < 0.01

# Article 13 — Transparency
ART13_MODEL_CARD_FIELDS = [
    "model_type",
    "training_date",
    "feature_list",
    "target",
    "intended_use",
    "limitations",
    "protected_attrs_documented",
]

# Article 14 — Human Oversight
ART14_BRIER_WARN = 0.20            # warn if Brier score > 0.20
ART14_BRIER_FAIL = 0.30            # fail if Brier score > 0.30
ART14_ECE_WARN = 0.10              # warn if ECE > 0.10
ART14_ECE_FAIL = 0.20              # fail if ECE > 0.20
ART14_LOW_CONF_THRESHOLD = 0.60    # flag rate: max_prob < 0.60 (informational)
ART14_ABSTENTION_THRESHOLD = 0.70  # abstention rate: max_prob < 0.70 (informational)

# Article 15 — Accuracy and Robustness
ART15_ACCURACY_WARN = 0.75         # warn if accuracy < 0.75
ART15_ACCURACY_FAIL = 0.65         # fail if accuracy < 0.65
ART15_FAIRNESS_WARN = 0.10         # warn if fairness metric > 0.10
ART15_FAIRNESS_FAIL = 0.20         # fail if fairness metric > 0.20
ART15_ROBUSTNESS_WARN = 0.85       # warn if prediction stability < 0.85
ART15_ROBUSTNESS_FAIL = 0.70       # fail if prediction stability < 0.70
ART15_NOISE_SIGMA = 0.05           # Gaussian noise std for robustness test
