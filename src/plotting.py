"""
plotting.py
-----------
Shared plotting functions used by the notebook (and importable by the app if
needed) so that figures are generated consistently and results are recorded
the same way every time this pipeline is run.

Every function returns the matplotlib Figure it drew on so the caller can
show it, save it, or embed it -- functions never call plt.show() themselves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve


def plot_rul_residuals(y_true, y_pred, title="RUL residuals"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(y_true, y_pred, s=10, alpha=0.4, color="steelblue")
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    axes[0].plot(lim, lim, color="red", linestyle="--", linewidth=1, label="perfect prediction")
    axes[0].set_xlabel("true RUL")
    axes[0].set_ylabel("predicted RUL")
    axes[0].set_title("Predicted vs true")
    axes[0].legend()

    residual = np.asarray(y_pred) - np.asarray(y_true)
    axes[1].scatter(y_true, residual, s=10, alpha=0.4, color="darkorange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("true RUL")
    axes[1].set_ylabel("residual (pred - true)")
    axes[1].set_title("Residuals vs true RUL")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_rul_trace(cycles, y_true, y_pred, engine_id=None, interval_std=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles, y_true, label="true RUL", color="black", linewidth=2)
    ax.plot(cycles, y_pred, label="predicted RUL", color="steelblue", linewidth=2)
    if interval_std is not None:
        ax.fill_between(
            cycles, np.array(y_pred) - 1.96 * interval_std, np.array(y_pred) + 1.96 * interval_std,
            color="steelblue", alpha=0.2, label="95% interval"
        )
    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL")
    title = f"Engine {engine_id} — RUL trace" if engine_id is not None else "RUL trace"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rul_trace_with_monotonicity(cycles, y_pred, engine_id=None, y_true=None):
    """Same as plot_rul_trace but marks every step where predicted RUL went
    UP instead of down with a red dot -- use this to visually sanity-check
    the "does RUL ever increase as cycle increases" question. A handful of
    small red dots is normal noise; many large jumps is a red flag (see
    src/diagnostics.engine_monotonicity_report for the numeric version of
    this same check)."""
    cycles = np.asarray(cycles)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(cycles)
    cycles, y_pred = cycles[order], y_pred[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    if y_true is not None:
        y_true = np.asarray(y_true)[order]
        ax.plot(cycles, y_true, label="true RUL", color="black", linewidth=2)
    ax.plot(cycles, y_pred, label="predicted RUL", color="steelblue", linewidth=1.5, marker="o", markersize=3)

    diffs = np.diff(y_pred)
    violation_idx = np.where(diffs > 0)[0] + 1  # index of the point where RUL went UP
    if len(violation_idx) > 0:
        ax.scatter(cycles[violation_idx], y_pred[violation_idx], color="red", zorder=5,
                   label=f"RUL increased here ({len(violation_idx)}x)", s=40)

    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL")
    title = f"Engine {engine_id} — monotonicity check" if engine_id is not None else "Monotonicity check"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_life_region_bars(region_report: pd.DataFrame, metric="MAE"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(region_report["region"], region_report[metric], color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by life region")
    for i, v in enumerate(region_report[metric]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true_dict: dict, proba_dict: dict, title="Precision-Recall"):
    """y_true_dict / proba_dict: {label: array} for overlaying multiple curves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in y_true_dict:
        y_true, proba = y_true_dict[label], proba_dict[label]
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(y_true, proba, n_bins=10, title="Calibration"):
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "not enough positive/negative\nsamples to calibrate", ha="center")
        return fig
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color="steelblue", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    brier = brier_score_loss(y_true, proba)
    ax.set_title(f"{title} (Brier={brier:.3f})")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_anomaly_histograms(scores_dict: dict, title="Anomaly score distributions (validation)"):
    """scores_dict: {detector_name: 1D array of scores}"""
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (name, scores) in zip(axes, scores_dict.items()):
        ax.hist(scores, bins=30, color="mediumpurple", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("score (larger = more abnormal)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_master_results_bar(master_results: pd.DataFrame, task_filter=None, title="Master results"):
    df = master_results if task_filter is None else master_results[master_results["task"] == task_filter]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(df))))
    y_pos = np.arange(len(df))
    err_low = df["point"] - df["ci_low"]
    err_high = df["ci_high"] - df["point"]
    ax.barh(y_pos, df["point"], xerr=[err_low, err_high], color="steelblue", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model"] + " (" + df["dataset"] + ")")
    ax.set_xlabel(df["metric"].iloc[0] if len(df) else "")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feat_cols, top_n=15, title="Feature importance"):
    if not hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.text(0.5, 0.5, "model has no feature_importances_", ha="center")
        return fig
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * top_n)))
    ax.barh(importances.index[::-1], importances.values[::-1], color="seagreen")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    return fig


def plot_lead_time(warn_df: pd.DataFrame, title="Early-warning lead time per engine"):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(warn_df))))
    colors = ["#C44E52" if m else "#55A868" for m in warn_df["missed"]]
    values = warn_df["lead_time"].fillna(0)
    ax.barh(warn_df["engine_id"].astype(str), values, color=colors)
    ax.set_xlabel("lead time (cycles)")
    ax.set_ylabel("engine_id")
    ax.set_title(title + "  (red = missed warning)")
    fig.tight_layout()
    return fig
