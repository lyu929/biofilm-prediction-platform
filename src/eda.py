from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .config import get_path, load_config
    from .feature_engineering import aggregate_cells_by_timepoint, aggregate_clusters_by_timepoint
    from .visualize import plot_cell_state_snapshot, plot_flow_field, plot_local_signal_heatmap, save_fig
except ImportError:
    from config import get_path, load_config
    from feature_engineering import aggregate_cells_by_timepoint, aggregate_clusters_by_timepoint
    from visualize import plot_cell_state_snapshot, plot_flow_field, plot_local_signal_heatmap, save_fig


SIMULATION_PARAMETERS = [
    "flow_rate",
    "adhesion_wall",
    "adhesion_cell",
    "diffusion_rate",
    "signal_decay",
    "qs_threshold",
    "division_rate",
    "eps_rate",
    "runtime",
    "random_seed",
]


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


def data_quality_report(data: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Create thesis-friendly data quality tables for all standard inputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, df in data.items():
        rows.append(
            {
                "table": name,
                "rows": int(len(df)),
                "columns": int(df.shape[1]) if not df.empty else 0,
                "missing_values": int(df.isna().sum().sum()) if not df.empty else 0,
                "duplicate_rows": int(df.duplicated().sum()) if not df.empty else 0,
                "simulation_ids": int(df["simulation_id"].nunique()) if "simulation_id" in df.columns else 0,
            }
        )
    table_report = pd.DataFrame(rows)
    table_report.to_csv(out / "data_quality_table_counts.csv", index=False)

    timepoint_report = pd.DataFrame()
    cells = data.get("cells", pd.DataFrame())
    if not cells.empty and {"simulation_id", "timepoint"}.issubset(cells.columns):
        timepoint_report = (
            cells.groupby("simulation_id")["timepoint"]
            .nunique()
            .rename("exported_timepoints")
            .reset_index()
        )
        timepoint_report.to_csv(out / "data_quality_timepoint_counts.csv", index=False)

    cell_count_report = pd.DataFrame()
    if not cells.empty and "simulation_id" in cells.columns:
        cell_count_report = (
            cells.groupby("simulation_id")
            .size()
            .rename("cell_records")
            .reset_index()
        )
        cell_count_report.to_csv(out / "data_quality_cell_record_counts.csv", index=False)

    duplicate_report = pd.DataFrame()
    if not cells.empty and {"cell_id", "simulation_id", "timepoint"}.issubset(cells.columns):
        dup_cols = ["cell_id", "simulation_id", "timepoint"]
        duplicate_report = (
            cells[cells.duplicated(dup_cols, keep=False)]
            .sort_values(dup_cols)
            .head(5000)
        )
        duplicate_report.to_csv(out / "data_quality_duplicate_cell_records.csv", index=False)

    outlier_rows = []
    for table, df in data.items():
        numeric = df.select_dtypes(include="number") if not df.empty else pd.DataFrame()
        for col in numeric.columns:
            s = pd.to_numeric(numeric[col], errors="coerce").dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outlier_rows.append(
                {
                    "table": table,
                    "column": col,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                    "iqr_outliers": int(((s < lo) | (s > hi)).sum()) if iqr > 0 else 0,
                }
            )
    outlier_report = pd.DataFrame(outlier_rows)
    outlier_report.to_csv(out / "data_quality_outliers.csv", index=False)

    return {
        "table_counts": table_report,
        "timepoint_counts": timepoint_report,
        "cell_record_counts": cell_count_report,
        "duplicate_cell_records": duplicate_report,
        "outliers": outlier_report,
    }


def _lineplot_over_time(df: pd.DataFrame, column: str, fig_dir: Path, ylabel: str | None = None) -> None:
    if df.empty or column not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df,
        x="timepoint",
        y=column,
        estimator="mean",
        errorbar=("ci", 95),
        color="#2f6fed",
    )
    plt.xlabel("Timepoint")
    plt.ylabel(ylabel or column)
    plt.title(f"{column} over time (mean with 95% CI)")
    save_fig(fig_dir / f"{column}_over_time.png")


def _parameter_sensitivity_plots(merged: pd.DataFrame, target: str, fig_dir: Path) -> None:
    for param in SIMULATION_PARAMETERS:
        if param not in merged.columns or target not in merged.columns:
            continue
        plot_df = merged[[param, target]].copy()
        plot_df[param] = pd.to_numeric(plot_df[param], errors="coerce")
        plot_df[target] = pd.to_numeric(plot_df[target], errors="coerce")
        plot_df = plot_df.dropna()
        if plot_df.empty:
            continue

        plt.figure(figsize=(5.5, 4))
        sns.scatterplot(data=plot_df, x=param, y=target, alpha=0.75, edgecolor=None)
        sns.regplot(data=plot_df, x=param, y=target, scatter=False, color="black", line_kws={"linewidth": 1})
        plt.xlabel(param)
        plt.ylabel(target)
        plt.title(f"{param} vs {target}")
        save_fig(fig_dir / f"{param}_vs_{target}.png")

        if plot_df[param].nunique() <= 12:
            plt.figure(figsize=(5.5, 4))
            sns.boxplot(data=plot_df, x=param, y=target, color="#9ecae1")
            sns.stripplot(data=plot_df, x=param, y=target, color="black", alpha=0.35, size=3)
            plt.xlabel(param)
            plt.ylabel(target)
            plt.title(f"{target} grouped by {param}")
            save_fig(fig_dir / f"{param}_boxplot_{target}.png")


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
    data_quality_report(data, result_dir)

    summary = data.get("simulation_summary", pd.DataFrame())
    simulations = data.get("simulations", pd.DataFrame())
    cells = data.get("cells", pd.DataFrame())
    clusters = data.get("clusters", pd.DataFrame())

    if not summary.empty and target in summary.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(pd.to_numeric(summary[target], errors="coerce").dropna(), kde=True, color="#2f6fed")
        plt.xlabel(target)
        plt.ylabel("Simulation count")
        plt.title(f"Distribution of {target}")
        save_fig(fig_dir / f"{target}_distribution.png")
    else:
        warnings.warn(f"Target '{target}' not available for EDA distribution plot.")

    if dataset is not None and not dataset.empty:
        numeric = dataset.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True)
            corr.to_csv(result_dir / "feature_correlation_matrix.csv")
            if target in corr.columns:
                ordered = corr[target].abs().sort_values(ascending=False)
                keep = [c for c in SIMULATION_PARAMETERS if c in corr.columns]
                keep += [c for c in ordered.index if c not in keep][:35]
                keep = list(dict.fromkeys(keep))
                corr_plot = corr.loc[keep, keep]
            else:
                corr_plot = corr.iloc[:40, :40]
            plt.figure(figsize=(11, 9))
            sns.heatmap(corr_plot, cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
            plt.title("Feature correlation heatmap")
            save_fig(fig_dir / "correlation_heatmap.png")

    if not simulations.empty and not summary.empty and target in summary.columns:
        merged = simulations.merge(summary[["simulation_id", target]], on="simulation_id", how="inner")
        _parameter_sensitivity_plots(merged, target, fig_dir)

    cell_agg = aggregate_cells_by_timepoint(cells)
    if not cell_agg.empty:
        cell_agg.to_csv(result_dir / "cell_timepoint_features.csv", index=False)
        for col in [
            "cell_count",
            "biofilm_thickness",
            "mean_volume",
            "attached_ratio",
            "qs_active_ratio",
            "eps_producing_ratio",
            "mean_local_signal",
        ]:
            _lineplot_over_time(cell_agg, col, fig_dir)

    cluster_agg = aggregate_clusters_by_timepoint(clusters)
    if not cluster_agg.empty:
        cluster_agg.to_csv(result_dir / "cluster_timepoint_features.csv", index=False)
        for col in ["cluster_count", "mean_cluster_size"]:
            _lineplot_over_time(cluster_agg, col, fig_dir)

    plot_cell_state_snapshot(cells, fig_dir / "biofilm_cell_state_snapshot.png")
    plot_local_signal_heatmap(cells, fig_dir / "local_signal_heatmap.png")
    plot_flow_field(data.get("flow_field", pd.DataFrame()), fig_dir / "flow_shear_field.png")


if __name__ == "__main__":
    from data_loader import load_all_data
    from dataset_builder import build_training_dataset

    loaded = load_all_data("csv")
    ds, _, _ = build_training_dataset(loaded, save=False)
    run_eda(loaded, ds)
