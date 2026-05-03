"""
消融实验（Ablation Study）
验证各类特征对预测的独立贡献：
  - params_only       : 仅仿真参数（10维）
  - params_flow       : 仿真参数 + 流场特征
  - params_cells      : 仿真参数 + 早期细胞时序特征
  - params_clusters   : 仿真参数 + 早期簇时序特征
  - all_features      : 全部特征（当前默认）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

try:
    from .config import get_path, load_config
    from .dataset_builder import load_processed_dataset
    from .feature_engineering import SIMULATION_FEATURES
except ImportError:
    from config import get_path, load_config
    from dataset_builder import load_processed_dataset
    from feature_engineering import SIMULATION_FEATURES

# 流场特征列名前缀
_FLOW_PREFIXES = (
    "mean_velocity", "std_velocity", "mean_speed",
    "max_speed", "mean_shear", "max_shear", "std_shear",
    "low_shear_fraction", "high_shear_fraction",
)

# 早期特征列前缀（t0_, t1_, …）
_EARLY_CELL_EXCLUDE = "cluster_"  # 细胞特征：含 t{n}_ 但不含 cluster_
_EARLY_CLUSTER_KEY = "cluster_"   # 簇特征：含 t{n}_cluster_


def _select_columns(X: pd.DataFrame, group: str) -> pd.DataFrame:
    """根据特征组名筛选列。"""
    all_cols = list(X.columns)

    sim_cols = [c for c in all_cols if c in SIMULATION_FEATURES]
    flow_cols = [c for c in all_cols
                 if any(c.startswith(p) for p in _FLOW_PREFIXES)]
    early_cell_cols = [c for c in all_cols
                       if c.startswith("t") and "_" in c
                       and _EARLY_CLUSTER_KEY not in c
                       and c not in sim_cols]
    early_cluster_cols = [c for c in all_cols
                          if c.startswith("t") and "_" in c
                          and _EARLY_CLUSTER_KEY in c]

    mapping = {
        "params_only": sim_cols,
        "params_flow": sim_cols + flow_cols,
        "params_cells": sim_cols + early_cell_cols,
        "params_clusters": sim_cols + early_cluster_cols,
        "all_features": all_cols,
    }
    cols = mapping.get(group, all_cols)
    # 保证列存在
    cols = [c for c in cols if c in X.columns]
    return X[cols] if cols else X


def _rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )),
    ])


def run_ablation(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    target: str = "biofilm_thickness",
    n_splits: int = 5,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    对每个特征组分别运行 K-fold 交叉验证，
    返回汇总 DataFrame 并保存图表。
    """
    cfg = load_config(config_path)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)

    if X is None or y is None:
        _, X, y = load_processed_dataset(config_path=config_path, target=target)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    groups = ["params_only", "params_flow", "params_cells", "params_clusters", "all_features"]
    rows = []

    for group in groups:
        X_sub = _select_columns(X, group)
        n_features = X_sub.shape[1]
        if n_features == 0:
            continue

        cv_result = cross_validate(
            _rf_pipeline(), X_sub, y,
            cv=kf,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )
        rows.append({
            "feature_group": group,
            "n_features": n_features,
            "target": target,
            "MAE_mean": -cv_result["test_mae"].mean(),
            "MAE_std": cv_result["test_mae"].std(),
            "RMSE_mean": -cv_result["test_rmse"].mean(),
            "RMSE_std": cv_result["test_rmse"].std(),
            "R2_mean": cv_result["test_r2"].mean(),
            "R2_std": cv_result["test_r2"].std(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(result_dir / f"ablation_{target}.csv", index=False)

    # ---------- 可视化 ----------
    _plot_ablation(df, target, fig_dir)

    return df


def _plot_ablation(df: pd.DataFrame, target: str, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels  = df["feature_group"].tolist()
    x       = np.arange(len(labels))
    bar_w   = 0.55
    n_feats = df["n_features"].tolist()

    # ── RMSE（只画上方误差棒，RMSE ≥ 0 下方无意义）─────────
    ax        = axes[0]
    rmse_mean = df["RMSE_mean"].values
    rmse_std  = df["RMSE_std"].values
    rmse_top  = float((rmse_mean + rmse_std).max()) * 1.20

    ax.bar(x, rmse_mean, bar_w,
           yerr=[np.zeros_like(rmse_std), rmse_std],
           capsize=5, color="#4C72B0", edgecolor="white",
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (mean ± std)", fontsize=10)
    ax.set_title(f"Ablation Study — RMSE\n{target}", fontsize=11)
    ax.set_ylim(0, max(rmse_top, 0.01))
    for i, (m, s, n) in enumerate(zip(rmse_mean, rmse_std, n_feats)):
        ax.text(x[i], m + s + rmse_top * 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)

    # ── R²（显示负值，双向误差棒）──────────────────────────
    ax       = axes[1]
    r2_mean  = df["R2_mean"].values
    r2_std   = df["R2_std"].values
    r2_floor = min(-0.05, float(r2_mean.min()) - 0.05)
    # yerr 必须 ≥ 0：下方距离截断到 [0, r2_mean - r2_floor]，上方截断到 1 - r2_mean
    r2_lo = np.clip(np.minimum(r2_std, r2_mean - r2_floor), 0, None)
    r2_hi = np.clip(np.minimum(r2_std, 1.0 - r2_mean),     0, None)

    ax.bar(x, r2_mean, bar_w,
           yerr=[r2_lo, r2_hi],
           capsize=5, color="#55A868", edgecolor="white",
           error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("R² (mean ± std)", fontsize=10)
    ax.set_title(f"Ablation Study — R²\n{target}", fontsize=11)
    ax.set_ylim(r2_floor, 1.05)
    span = 1.05 - r2_floor
    for i, (m, hi, n) in enumerate(zip(r2_mean, r2_hi, n_feats)):
        y_text = max(m, r2_floor) + hi + span * 0.02
        ax.text(x[i], min(y_text, 1.03),
                f"n={n}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / f"ablation_{target}.png", dpi=150)
    plt.close()


def run_ablation_all_targets(
    X: pd.DataFrame | None = None,
    y_all: dict[str, pd.Series] | None = None,
    config_path: str | Path | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """对所有预测目标分别运行消融实验，汇总结果。"""
    try:
        from .config import TARGET_COLUMNS
        from .dataset_builder import load_processed_dataset
    except ImportError:
        from config import TARGET_COLUMNS
        from dataset_builder import load_processed_dataset

    cfg = load_config(config_path)

    if X is None:
        _, X, _ = load_processed_dataset(config_path=config_path)

    all_rows = []
    for target in TARGET_COLUMNS:
        try:
            _, _, y = load_processed_dataset(config_path=config_path, target=target)
            df = run_ablation(X, y, target=target, n_splits=n_splits,
                              config_path=config_path)
            all_rows.append(df)
        except Exception as exc:
            print(f"[ablation] Skipping {target}: {exc}")

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result_dir = get_path("results", cfg)
    result.to_csv(result_dir / "ablation_all_targets.csv", index=False)
    return result


if __name__ == "__main__":
    df = run_ablation()
    print(df.to_string(index=False))
