from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .config import get_path, load_config
    from .feature_engineering import aggregate_cells_by_timepoint, aggregate_clusters_by_timepoint
    from .visualize import save_fig
except ImportError:
    from config import get_path, load_config
    from feature_engineering import aggregate_cells_by_timepoint, aggregate_clusters_by_timepoint
    from visualize import save_fig


def missing_value_report(data: dict[str, pd.DataFrame], output_path: str | Path | None = None) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        if df.empty:
            rows.append({"table": name, "column": "__empty__", "missing_count": 0, "missing_fraction": 0.0})
            continue
        for col in df.columns:
            rows.append(
                {
                    "table": name,
                    "column": col,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_fraction": float(df[col].isna().mean()),
                }
            )
    report = pd.DataFrame(rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)
    return report


def run_eda(
    data: dict[str, pd.DataFrame],
    dataset: pd.DataFrame | None = None,
    target: str = "biofilm_thickness",
    config_path: str | Path | None = None,
) -> None:
    cfg = load_config(config_path)
    fig_dir = get_path("figures", cfg)
    result_dir = get_path("results", cfg)
    missing_value_report(data, result_dir / "missing_values.csv")

    summary = data.get("simulation_summary", pd.DataFrame())
    simulations = data.get("simulations", pd.DataFrame())
    cells = data.get("cells", pd.DataFrame())
    clusters = data.get("clusters", pd.DataFrame())

    if not summary.empty and target in summary.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(summary[target], kde=True)
        plt.title(f"{target} distribution")
        save_fig(fig_dir / f"{target}_distribution.png")
    else:
        warnings.warn(f"Target '{target}' not available for EDA distribution plot.")

    if dataset is not None and not dataset.empty:
        numeric = dataset.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            corr = numeric.corr(numeric_only=True)
            sns.heatmap(corr, cmap="vlag", center=0)
            plt.title("Feature correlation heatmap")
            save_fig(fig_dir / "correlation_heatmap.png")

    if not simulations.empty and not summary.empty and target in summary.columns:
        merged = simulations.merge(summary[["simulation_id", target]], on="simulation_id", how="inner")
        params = [
            "flow_rate",
            "adhesion_wall",
            "adhesion_cell",
            "diffusion_rate",
            "signal_decay",
            "qs_threshold",
            "division_rate",
            "eps_rate",
        ]
        for param in params:
            if param in merged.columns:
                plt.figure(figsize=(5, 4))
                sns.scatterplot(data=merged, x=param, y=target)
                plt.title(f"{param} vs {target}")
                save_fig(fig_dir / f"{param}_vs_{target}.png")

    cell_agg = aggregate_cells_by_timepoint(cells)
    if not cell_agg.empty:
        for col in ["cell_count", "mean_volume", "attached_ratio"]:
            if col in cell_agg.columns:
                plt.figure(figsize=(7, 4))
                sns.lineplot(data=cell_agg, x="timepoint", y=col, estimator="mean", errorbar=None)
                plt.title(f"{col} over time")
                save_fig(fig_dir / f"{col}_over_time.png")

    cluster_agg = aggregate_clusters_by_timepoint(clusters)
    if not cluster_agg.empty:
        for col in ["cluster_count", "mean_cluster_size"]:
            if col in cluster_agg.columns:
                plt.figure(figsize=(7, 4))
                sns.lineplot(data=cluster_agg, x="timepoint", y=col, estimator="mean", errorbar=None)
                plt.title(f"{col} over time")
                save_fig(fig_dir / f"{col}_over_time.png")


if __name__ == "__main__":
    from data_loader import load_all_data
    from dataset_builder import build_training_dataset

    loaded = load_all_data("csv")
    ds, _, _ = build_training_dataset(loaded, save=False)
    run_eda(loaded, ds)
