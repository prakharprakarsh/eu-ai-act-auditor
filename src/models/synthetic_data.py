from __future__ import annotations

import numpy as np
import pandas as pd


def generate_hiring_dataset(
    n: int = 5000,
    seed: int = 42,
    bias_strength: float = 0.25,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)

    years_experience = rng.gamma(shape=2.0, scale=4.0, size=n).clip(0, 25)
    education_level = rng.choice([0, 1, 2, 3], size=n, p=[0.15, 0.45, 0.30, 0.10])
    prior_role_match = rng.beta(a=2.5, b=2.0, size=n).clip(0, 1)
    skills_overlap = rng.beta(a=2.0, b=2.0, size=n).clip(0, 1)
    gap_months = rng.exponential(scale=6.0, size=n).clip(0, 36)
    gender = rng.integers(0, 2, size=n)
    age_band = rng.choice([0, 1, 2], size=n, p=[0.35, 0.45, 0.20])
    ethnicity_proxy = rng.integers(0, 5, size=n)

    log_odds = (
        0.25 * years_experience
        + 0.60 * education_level
        + 1.80 * prior_role_match
        + 1.60 * skills_overlap
        - 0.03 * gap_months
        - 2.50
    )

    log_odds -= bias_strength * 3.5 * (gender == 1)
    log_odds -= bias_strength * 3.0 * (age_band == 2)

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    hired = rng.binomial(n=1, p=prob).astype(int)

    X = pd.DataFrame(
        {
            "years_experience": years_experience.astype(np.float32),
            "education_level": education_level.astype(np.int8),
            "prior_role_match": prior_role_match.astype(np.float32),
            "skills_overlap": skills_overlap.astype(np.float32),
            "gap_months": gap_months.astype(np.float32),
            "gender": gender.astype(np.int8),
            "age_band": age_band.astype(np.int8),
            "ethnicity_proxy": ethnicity_proxy.astype(np.int8),
        }
    )
    y = pd.Series(hired, name="hired", dtype=np.int8)
    return X, y
