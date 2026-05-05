from __future__ import annotations

import numpy as np

from auditor.base import AuditStatus


def threshold_status(
    value: float,
    warn: float,
    fail: float,
    higher_is_worse: bool = True,
) -> AuditStatus:
    if higher_is_worse:
        if value > fail:
            return AuditStatus.FAIL
        if value > warn:
            return AuditStatus.WARN
    else:
        if value < fail:
            return AuditStatus.FAIL
        if value < warn:
            return AuditStatus.WARN
    return AuditStatus.PASS


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Last bin is right-inclusive so prob=1.0 is never dropped.
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)
