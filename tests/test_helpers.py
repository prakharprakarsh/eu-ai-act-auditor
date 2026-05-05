import numpy as np
import pytest

from auditor._helpers import compute_ece, threshold_status
from auditor.base import AuditStatus


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

def _balanced(n: int = 1000) -> np.ndarray:
    """Balanced binary labels: first half 0, second half 1."""
    return np.array([0.0] * (n // 2) + [1.0] * (n // 2))


def test_ece_perfectly_calibrated_is_zero():
    # Predict exactly 0.0 for class 0, 1.0 for class 1.
    # Every bin has avg_confidence == fraction_positive → ECE = 0.
    n = 1000
    y_true = _balanced(n)
    y_prob = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
    assert compute_ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)


def test_ece_perfectly_inverted_is_one():
    # Predict 1.0 for class 0, 0.0 for class 1.
    # Every bin is maximally miscalibrated → ECE = 1.0.
    # Bug: without the right-inclusive last bin, prob=1.0 samples are dropped
    # and ECE collapses to 0.5.
    n = 1000
    y_true = _balanced(n)
    y_prob = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
    assert compute_ece(y_true, y_prob) == pytest.approx(1.0, abs=1e-9)


def test_ece_random_predictions_near_half():
    # Predictions drawn from {0, 1} uniformly at random, independent of labels.
    # Each extreme bin contains ~50 % positives: |0.5 - conf| = 0.5 for both bins.
    # ECE = weight_low * 0.5 + weight_high * 0.5 ≈ 0.5.
    n = 2000
    y_true = _balanced(n)
    rng = np.random.default_rng(0)
    y_prob = rng.choice([0.0, 1.0], size=n)
    ece = compute_ece(y_true, y_prob)
    assert 0.35 < ece < 0.65, f"Expected ECE ≈ 0.5 for random predictions, got {ece:.4f}"


def test_ece_prob_one_not_dropped():
    # Regression: a single sample with prob=1.0 must not be silently excluded.
    y_true = np.array([1.0])
    y_prob = np.array([1.0])
    # acc=1, conf=1 → gap=0 → ECE=0
    assert compute_ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)


def test_ece_high_confidence_correct_predictions_low():
    # Predict 0.1 for class 0, 0.9 for class 1 (strong discrimination, slight
    # miscalibration of 0.1 per bin) → ECE = 0.1, well below random-guess level.
    n = 1000
    y_true = _balanced(n)
    y_prob = np.array([0.1] * (n // 2) + [0.9] * (n // 2))
    ece = compute_ece(y_true, y_prob)
    assert ece == pytest.approx(0.1, abs=1e-9)


def test_ece_returns_float():
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    assert isinstance(compute_ece(y_true, y_prob), float)


def test_ece_all_same_prob_balanced():
    # All predictions = 0.5, balanced labels → acc=0.5, conf=0.5, ECE=0.
    n = 100
    y_true = _balanced(n)
    y_prob = np.full(n, 0.5)
    assert compute_ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# threshold_status
# ---------------------------------------------------------------------------

def test_threshold_status_pass_higher_is_worse():
    assert threshold_status(0.05, warn=0.10, fail=0.20) == AuditStatus.PASS


def test_threshold_status_warn_higher_is_worse():
    assert threshold_status(0.15, warn=0.10, fail=0.20) == AuditStatus.WARN


def test_threshold_status_fail_higher_is_worse():
    assert threshold_status(0.25, warn=0.10, fail=0.20) == AuditStatus.FAIL


def test_threshold_status_pass_lower_is_worse():
    assert threshold_status(0.80, warn=0.75, fail=0.65, higher_is_worse=False) == AuditStatus.PASS


def test_threshold_status_warn_lower_is_worse():
    assert threshold_status(0.70, warn=0.75, fail=0.65, higher_is_worse=False) == AuditStatus.WARN


def test_threshold_status_fail_lower_is_worse():
    assert threshold_status(0.60, warn=0.75, fail=0.65, higher_is_worse=False) == AuditStatus.FAIL


def test_threshold_status_exactly_at_warn_boundary_higher_is_worse():
    # value == warn: higher_is_worse uses strict >, so exactly at boundary → PASS
    assert threshold_status(0.10, warn=0.10, fail=0.20) == AuditStatus.PASS


def test_threshold_status_exactly_at_warn_boundary_lower_is_worse():
    # value == warn: lower_is_worse uses strict <, so exactly at boundary → PASS
    assert threshold_status(0.75, warn=0.75, fail=0.65, higher_is_worse=False) == AuditStatus.PASS
