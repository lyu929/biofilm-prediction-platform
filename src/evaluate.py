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


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "RMSE",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap 置信区间估计。
    返回 (point_estimate, lower_bound, upper_bound)。
    metric 可选: 'MAE', 'RMSE', 'R2'
    """
    rng = np.random.default_rng(random_seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    def _calc(yt, yp):
        if metric == "MAE":
            return float(mean_absolute_error(yt, yp))
        if metric == "RMSE":
            return float(np.sqrt(mean_squared_error(yt, yp)))
        if metric == "R2":
            return float(r2_score(yt, yp)) if len(yt) > 1 else 0.0
        raise ValueError(f"Unknown metric: {metric}")

    point = _calc(y_true, y_pred)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores.append(_calc(y_true[idx], y_pred[idx]))

    alpha = 1 - ci
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point, lower, upper


def regression_metrics_with_ci(
    y_true,
    y_pred,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    """计算 MAE / RMSE / R² 及其 Bootstrap 95% 置信区间。"""
    result = {}
    for metric in ("MAE", "RMSE", "R2"):
        point, lower, upper = bootstrap_ci(
            y_true, y_pred, metric=metric,
            n_bootstrap=n_bootstrap, ci=ci, random_seed=random_seed,
        )
        result[metric] = point
        result[f"{metric}_CI_lower"] = lower
        result[f"{metric}_CI_upper"] = upper
    return result


def plot_metrics_with_ci(
    metrics_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """绘制含误差棒的模型对比图（来自 regression_metrics_with_ci 的结果）。"""
    import matplotlib.pyplot as plt

    col_lower = f"{metric}_CI_lower"
    col_upper = f"{metric}_CI_upper"
    if col_lower not in metrics_df.columns or col_upper not in metrics_df.columns:
        return

    models = metrics_df["model"].tolist()
    vals = metrics_df[metric].tolist()
    errs_low = [v - metrics_df[col_lower].iloc[i] for i, v in enumerate(vals)]
    errs_high = [metrics_df[col_upper].iloc[i] - v for i, v in enumerate(vals)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, vals,
           yerr=[errs_low, errs_high],
           capsize=5, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} with 95% Bootstrap CI")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str,
    target: str,
    config_path: str | Path | None = None,
    with_ci: bool = True,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)

    if with_ci:
        metrics = regression_metrics_with_ci(y_true, y_pred)
    else:
        metrics = regression_metrics(y_true, y_pred)

    df = pd.DataFrame([{**{"model": model_name, "target": target}, **metrics}])
    df.to_csv(result_dir / f"{model_name}_{target}_metrics.csv", index=False)
    plot_pred_vs_true(
        y_true, y_pred,
        f"{model_name}: predicted vs true",
        fig_dir / f"{model_name}_{target}_pred_vs_true.png",
    )
    plot_residuals(
        y_true, y_pred,
        f"{model_name}: residuals",
        fig_dir / f"{model_name}_{target}_residuals.png",
    )
    return df
