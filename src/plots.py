from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_dir: str | Path,
    prefix: str = "test",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residuals = y_true - y_pred

    sns.set_theme(style="whitegrid")

    # Pred vs actual
    plt.figure(figsize=(7, 6))
    ax = sns.scatterplot(x=y_true, y=y_pred, s=15, alpha=0.35)
    lim_min = float(min(y_true.min(), y_pred.min()))
    lim_max = float(max(y_true.max(), y_pred.max()))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1)
    ax.set_title("Predicted vs Actual (Price)")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pred_vs_actual.png", dpi=200)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(7, 4.5))
    sns.histplot(residuals, bins=60, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_residuals_hist.png", dpi=200)
    plt.close()

    # Residuals vs predicted
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(x=y_pred, y=residuals, s=15, alpha=0.35)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_residuals_vs_pred.png", dpi=200)
    plt.close()
