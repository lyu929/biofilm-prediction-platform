from __future__ import annotations

from typing import Iterable
import warnings

import numpy as np
import pandas as pd

try:
    from .config import warn_missing_columns
except ImportError:
    from config import warn_missing_columns


SIMULATION_FEATURES = [
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

CELL_STATES = [
    "active",
    "inactive",
    "attached",
    "qs_active",
    "eps_producing",
    "planktonic",
    "detached",
]


def _safe_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def extract_simulation_features(simulations: pd.DataFrame) -> pd.DataFrame:
    if simulations.empty:
        return pd.DataFrame(columns=["simulation_id"] + SIMULATION_FEATURES)
    warn_missing_columns(simulations, ["simulation_id"] + SIMULATION_FEATURES, "simulations")
    out = simulations.copy()
    if "simulation_id" not in out.columns:
        out["simulation_id"] = [f"simulation_{i}" for i in range(len(out))]
    for col in SIMULATION_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    out = _safe_numeric(out, SIMULATION_FEATURES)
    return out[["simulation_id"] + SIMULATION_FEATURES].drop_duplicates("simulation_id")


def aggregate_cells_by_timepoint(cells: pd.DataFrame) -> pd.DataFrame:
    required = ["simulation_id", "timepoint", "x", "y", "state", "volume", "local_signal"]
    if cells.empty:
        return pd.DataFrame(columns=["simulation_id", "timepoint"])
    warn_missing_columns(cells, required, "cells")
    df = cells.copy()
    for col in required:
        if col not in df.columns:
            df[col] = 0 if col not in {"simulation_id", "state"} else ""
    df = _safe_numeric(df, ["timepoint", "x", "y", "volume", "local_signal"])
    df["state"] = df["state"].fillna("unknown").astype(str)

    grouped = df.groupby(["simulation_id", "timepoint"], dropna=False)
    agg = grouped.agg(
        cell_count=("cell_id", "count") if "cell_id" in df.columns else ("state", "count"),
        mean_x=("x", "mean"),
        mean_y=("y", "mean"),
        std_x=("x", "std"),
        std_y=("y", "std"),
        mean_volume=("volume", "mean"),
        std_volume=("volume", "std"),
        mean_local_signal=("local_signal", "mean"),
        std_local_signal=("local_signal", "std"),
    ).reset_index()

    state_counts = (
        df.groupby(["simulation_id", "timepoint", "state"]).size().rename("n").reset_index()
    )
    totals = state_counts.groupby(["simulation_id", "timepoint"])["n"].transform("sum")
    state_counts["ratio"] = np.where(totals > 0, state_counts["n"] / totals, 0.0)
    ratios = state_counts.pivot_table(
        index=["simulation_id", "timepoint"],
        columns="state",
        values="ratio",
        fill_value=0.0,
    ).reset_index()
    ratios.columns = [str(c) for c in ratios.columns]
    for state in CELL_STATES:
        if state not in ratios.columns:
            ratios[state] = 0.0
    rename = {state: f"{state}_ratio" for state in CELL_STATES if state != "planktonic"}
    ratios = ratios.rename(columns=rename)

    out = agg.merge(ratios[["simulation_id", "timepoint"] + list(rename.values())], on=["simulation_id", "timepoint"], how="left")
    return out.fillna(0.0)


def aggregate_clusters_by_timepoint(clusters: pd.DataFrame) -> pd.DataFrame:
    required = ["simulation_id", "timepoint", "cluster_size", "center_x", "center_y", "density"]
    if clusters.empty:
        return pd.DataFrame(columns=["simulation_id", "timepoint"])
    warn_missing_columns(clusters, required, "clusters")
    df = clusters.copy()
    for col in required:
        if col not in df.columns:
            df[col] = 0 if col != "simulation_id" else ""
    df = _safe_numeric(df, ["timepoint", "cluster_size", "center_x", "center_y", "density"])
    grouped = df.groupby(["simulation_id", "timepoint"], dropna=False)
    out = grouped.agg(
        cluster_count=("cluster_id", "count") if "cluster_id" in df.columns else ("cluster_size", "count"),
        mean_cluster_size=("cluster_size", "mean"),
        max_cluster_size=("cluster_size", "max"),
        std_cluster_size=("cluster_size", "std"),
        mean_density=("density", "mean"),
        std_density=("density", "std"),
        mean_center_x=("center_x", "mean"),
        mean_center_y=("center_y", "mean"),
        cluster_spread_x=("center_x", "std"),
        cluster_spread_y=("center_y", "std"),
    ).reset_index()
    return out.fillna(0.0)


def aggregate_flow_field(flow_field: pd.DataFrame) -> pd.DataFrame:
    required = ["simulation_id", "velocity_x", "velocity_y", "shear"]
    if flow_field.empty:
        return pd.DataFrame(columns=["simulation_id"])
    warn_missing_columns(flow_field, required, "flow_field")
    df = flow_field.copy()
    for col in required:
        if col not in df.columns:
            df[col] = 0 if col != "simulation_id" else ""
    df = _safe_numeric(df, ["velocity_x", "velocity_y", "shear"])
    df["speed"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2)

    base = df.groupby("simulation_id").agg(
        mean_velocity_x=("velocity_x", "mean"),
        mean_velocity_y=("velocity_y", "mean"),
        std_velocity_x=("velocity_x", "std"),
        std_velocity_y=("velocity_y", "std"),
        mean_speed=("speed", "mean"),
        max_speed=("speed", "max"),
        mean_shear=("shear", "mean"),
        max_shear=("shear", "max"),
        std_shear=("shear", "std"),
    ).reset_index()

    fractions = []
    for sid, group in df.groupby("simulation_id"):
        q25 = group["shear"].quantile(0.25)
        q75 = group["shear"].quantile(0.75)
        fractions.append(
            {
                "simulation_id": sid,
                "low_shear_fraction": float((group["shear"] < q25).mean()),
                "high_shear_fraction": float((group["shear"] > q75).mean()),
            }
        )
    return base.merge(pd.DataFrame(fractions), on="simulation_id", how="left").fillna(0.0)


def _early_pivot(time_df: pd.DataFrame, early_timepoints: int, prefix: str) -> pd.DataFrame:
    if time_df.empty or "simulation_id" not in time_df.columns or "timepoint" not in time_df.columns:
        return pd.DataFrame(columns=["simulation_id"])
    feature_cols = [c for c in time_df.columns if c not in {"simulation_id", "timepoint"}]
    frames = []
    for sid, group in time_df.sort_values(["simulation_id", "timepoint"]).groupby("simulation_id"):
        group = group.head(early_timepoints).reset_index(drop=True)
        row = {"simulation_id": sid}
        for idx, rec in group.iterrows():
            for col in feature_cols:
                row[f"t{idx}_{prefix}{col}"] = rec[col]
        frames.append(row)
    return pd.DataFrame(frames).fillna(0.0)


def extract_early_timepoint_features(
    cells: pd.DataFrame,
    clusters: pd.DataFrame,
    early_timepoints: int = 10,
) -> pd.DataFrame:
    cell_agg = aggregate_cells_by_timepoint(cells)
    cluster_agg = aggregate_clusters_by_timepoint(clusters)
    cell_features = _early_pivot(cell_agg, early_timepoints, "")
    cluster_features = _early_pivot(cluster_agg, early_timepoints, "cluster_")
    if cell_features.empty:
        return cluster_features
    if cluster_features.empty:
        return cell_features
    return cell_features.merge(cluster_features, on="simulation_id", how="outer").fillna(0.0)


def build_feature_table(data: dict[str, pd.DataFrame], early_timepoints: int = 10) -> pd.DataFrame:
    simulations = data.get("simulations", pd.DataFrame())
    cells = data.get("cells", pd.DataFrame())
    clusters = data.get("clusters", pd.DataFrame())
    flow = data.get("flow_field", pd.DataFrame())

    features = extract_simulation_features(simulations)
    early = extract_early_timepoint_features(cells, clusters, early_timepoints=early_timepoints)
    flow_features = aggregate_flow_field(flow)

    if features.empty:
        warnings.warn("No simulations table available. Feature table may be empty.")
        return pd.DataFrame()
    for df in [early, flow_features]:
        if not df.empty and "simulation_id" in df.columns:
            features = features.merge(df, on="simulation_id", how="left")
    return features.fillna(0.0)
