from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ACCENT = "#1F4E79"
_WARN_LINE = "#E65100"
_FAIL_LINE = "#C62828"


def _ax_clean(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCCCCC")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#111111")
    ax.title.set_fontsize(10)


def _save(fig: Figure, tmpdir: Path, name: str) -> Path:
    path = tmpdir / name
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    return path


def plot_class_balance(class_dist: dict, tmpdir: Path) -> Path:
    fig = Figure(figsize=(3, 2), facecolor="white")
    ax = fig.add_subplot(111)
    _ax_clean(ax)

    sorted_items = sorted(class_dist.items(), key=lambda kv: kv[0])
    labels = [str(k) for k, _ in sorted_items]
    vals = [float(v) for _, v in sorted_items]
    bars = ax.bar(labels, vals, color=ACCENT, alpha=0.85, width=0.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Fraction")
    ax.set_title("Class Distribution")
    ax.set_ylim(0, 1.15)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.03,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    return _save(fig, tmpdir, "class_balance.png")


def plot_protected_coverage(coverage: dict[str, dict], tmpdir: Path) -> Path:
    attrs = list(coverage)
    n = len(attrs)
    fig = Figure(figsize=(max(4, 2.5 * n), 2.2), facecolor="white")
    fig.patch.set_facecolor("white")

    for i, attr in enumerate(attrs, 1):
        ax = fig.add_subplot(1, n, i)
        _ax_clean(ax)
        groups = coverage[attr]
        g_labels = [str(k) for k in groups]
        g_vals = [float(v) for v in groups.values()]
        ax.bar(g_labels, g_vals, color=ACCENT, alpha=0.85, width=0.5)
        ax.set_title(attr, fontsize=9)
        ax.set_ylabel("Fraction" if i == 1 else "")
        ax.set_ylim(0, 1.15)

    fig.suptitle("Protected Attribute Group Coverage", fontsize=10, color="#111111", y=1.04)
    return _save(fig, tmpdir, "protected_coverage.png")


def plot_shap_importance(top_features: list[tuple[str, float]], tmpdir: Path) -> Path:
    fig = Figure(figsize=(6, 3.5), facecolor="white")
    ax = fig.add_subplot(111)
    _ax_clean(ax)

    names = [f[0] for f in top_features]
    vals = [float(f[1]) for f in top_features]
    y = list(range(len(names)))
    ax.barh(y, vals, color=ACCENT, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    ax.invert_yaxis()
    return _save(fig, tmpdir, "shap_importance.png")


def plot_calibration_curve(
    y_true,
    y_prob,
    n_bins: int = 10,
    tmpdir: Path | None = None,
) -> Path:
    if tmpdir is None:
        tmpdir = Path(tempfile.mkdtemp())

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_fracs = [], []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_p >= lo) & (y_p <= hi if i == n_bins - 1 else y_p < hi)
        if not mask.any():
            continue
        bin_confs.append(float(y_p[mask].mean()))
        bin_fracs.append(float(y_t[mask].mean()))

    fig = Figure(figsize=(5, 4), facecolor="white")
    ax = fig.add_subplot(111)
    _ax_clean(ax)
    ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1.5,
            label="Perfect calibration")
    if bin_confs:
        ax.plot(bin_confs, bin_fracs, "o-", color=ACCENT, linewidth=2,
                markersize=5, label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, frameon=False)
    return _save(fig, tmpdir, "calibration_curve.png")


def plot_fairness_metrics(metrics: dict[str, dict], tmpdir: Path) -> Path:
    attrs = list(metrics)
    x = np.arange(len(attrs))
    dpd = [abs(float(metrics[a].get("dpd", 0))) for a in attrs]
    eod = [abs(float(metrics[a].get("eod", 0))) for a in attrs]
    w = 0.35

    fig = Figure(figsize=(max(5, 2.5 * len(attrs)), 4), facecolor="white")
    ax = fig.add_subplot(111)
    _ax_clean(ax)
    ax.bar(x - w / 2, dpd, w, label="Demographic Parity Diff",
           color=ACCENT, alpha=0.85)
    ax.bar(x + w / 2, eod, w, label="Equalized Odds Diff",
           color="#C77B30", alpha=0.85)
    ax.axhline(0.10, color=_WARN_LINE, linestyle="--", linewidth=1.2, label="Warn ≥ 0.10")
    ax.axhline(0.20, color=_FAIL_LINE, linestyle="--", linewidth=1.2, label="Fail ≥ 0.20")
    ax.set_xticks(list(x))
    ax.set_xticklabels(attrs, fontsize=9)
    ax.set_ylabel("Absolute Difference")
    ax.set_title("Fairness Metrics by Protected Attribute")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, tmpdir, "fairness_metrics.png")


def plot_shap_local_samples(samples: dict, tmpdir: Path) -> Path:
    """1 × N grid of horizontal bar charts — one per sample, top-6 features each."""
    n = len(samples)
    if n == 0:
        fig = Figure(figsize=(6, 2), facecolor="white")
        return _save(fig, tmpdir, "shap_local.png")

    fig = Figure(figsize=(min(14, 2.8 * n), 3.5), facecolor="white")
    fig.patch.set_facecolor("white")

    for col_idx, (sample_name, feats) in enumerate(samples.items(), 1):
        ax = fig.add_subplot(1, n, col_idx)
        _ax_clean(ax)

        top = sorted(feats.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        names = [k for k, _ in top]
        vals = [v for _, v in top]
        bar_colors = [ACCENT if v >= 0 else "#C62828" for v in vals]

        y = list(range(len(names)))
        ax.barh(y, vals, color=bar_colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names if col_idx == 1 else [""] * len(names), fontsize=7)
        ax.axvline(0, color="#CCCCCC", linewidth=0.8)
        ax.set_title(f"S{col_idx}", fontsize=8)
        ax.invert_yaxis()

    fig.suptitle("Local SHAP Explanations (sample contributions)", fontsize=9,
                 color="#111111", y=1.02)
    return _save(fig, tmpdir, "shap_local.png")
