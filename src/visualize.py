from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STATE_COLORS = {
    "planktonic": "#d3d3d3",
    "attached": "#808080",
    "active": "#2f6fed",
    "inactive": "#0b2a5b",
    "qs_active": "#d62728",
    "eps_producing": "#7b3294",
    "detached": "#bdbdbd",
}


def save_fig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_pred_vs_true(y_true, y_pred, title: str, output_path: str | Path) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    lo = float(min(y_true.min(), y_pred.min())) if len(y_true) else 0.0
    hi = float(max(y_true.max(), y_pred.max())) if len(y_true) else 1.0
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    save_fig(output_path)


def plot_residuals(y_true, y_pred, title: str, output_path: str | Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    save_fig(output_path)


def plot_feature_importance(importances: pd.Series, output_path: str | Path, top_n: int = 20) -> None:
    s = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, 0.28 * len(s))))
    sns.barplot(x=s.values, y=s.index)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    save_fig(output_path)
