from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import get_path, load_config
    from .visualize import plot_pred_vs_true, plot_residuals
except ImportError:
    from config import get_path, load_config
    from visualize import plot_pred_vs_true, plot_residuals


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }


def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str,
    target: str,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)
    metrics = regression_metrics(y_true, y_pred)
    df = pd.DataFrame([{**{"model": model_name, "target": target}, **metrics}])
    df.to_csv(result_dir / f"{model_name}_{target}_metrics.csv", index=False)
    plot_pred_vs_true(y_true, y_pred, f"{model_name}: predicted vs true", fig_dir / f"{model_name}_{target}_pred_vs_true.png")
    plot_residuals(y_true, y_pred, f"{model_name}: residuals", fig_dir / f"{model_name}_{target}_residuals.png")
    return df
