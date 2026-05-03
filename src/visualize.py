from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STATE_COLORS = {
    "planktonic": "#d3d3d3",      # light gray
    "attached": "#808080",        # gray
    "active": "#2f6fed",          # blue
    "inactive": "#0b2a5b",        # dark blue
    "qs_active": "#d62728",       # red
    "eps_producing": "#7b3294",   # purple
    "detached": "#bdbdbd",
}
EPS_MATRIX_COLOR = "#2ca25f"


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
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0)
        lo -= pad
        hi += pad
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    save_fig(output_path)


def plot_residuals(y_true, y_pred, title: str, output_path: str | Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted value")
    plt.ylabel("Residual (true - predicted)")
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


def _choose_snapshot(df: pd.DataFrame, simulation_id: str | None, timepoint: int | None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if simulation_id is None and "simulation_id" in out.columns:
        simulation_id = str(out["simulation_id"].dropna().astype(str).iloc[0])
    if simulation_id is not None and "simulation_id" in out.columns:
        out = out[out["simulation_id"].astype(str) == str(simulation_id)]
    if timepoint is None and "timepoint" in out.columns and not out.empty:
        timepoint = int(pd.to_numeric(out["timepoint"], errors="coerce").max())
    if timepoint is not None and "timepoint" in out.columns:
        out = out[pd.to_numeric(out["timepoint"], errors="coerce") == timepoint]
    return out


def plot_cell_state_snapshot(
    cells: pd.DataFrame,
    output_path: str | Path,
    simulation_id: str | None = None,
    timepoint: int | None = None,
    title: str | None = None,
) -> None:
    """Plot one exported cell snapshot with y converted to distance from wall.

    Exported simulations store canvas-style coordinates where the wall is at
    high y. For analysis figures, this plot converts to distance-from-wall so
    the wall is the x-axis and biofilm height grows upward.
    """
    required = {"x", "y", "state"}
    if cells.empty or not required.issubset(cells.columns):
        return
    snap = _choose_snapshot(cells, simulation_id, timepoint)
    if snap.empty:
        return
    snap = snap.copy()
    snap["x"] = pd.to_numeric(snap["x"], errors="coerce")
    snap["y"] = pd.to_numeric(snap["y"], errors="coerce")
    snap = snap.dropna(subset=["x", "y"])
    if snap.empty:
        return
    wall_y = float(pd.to_numeric(cells["y"], errors="coerce").max())
    snap["distance_from_wall"] = np.maximum(0.0, wall_y - snap["y"])
    sizes = pd.to_numeric(snap.get("volume", pd.Series(12, index=snap.index)), errors="coerce").fillna(12)
    sizes = np.clip(sizes, 8, 80)

    plt.figure(figsize=(8, 4))
    for state, group in snap.groupby(snap["state"].fillna("unknown").astype(str)):
        plt.scatter(
            group["x"],
            group["distance_from_wall"],
            s=sizes.loc[group.index],
            c=STATE_COLORS.get(state, "#999999"),
            label=state,
            alpha=0.82,
            linewidths=0.2,
            edgecolors="black",
        )
    plt.axhline(0, color="black", linewidth=1.2, label="wall")
    plt.xlabel("x position along wall")
    plt.ylabel("distance from wall")
    plt.title(title or "Biofilm cell-state spatial distribution")
    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.gca().set_aspect("equal", adjustable="box")
    save_fig(output_path)


def plot_local_signal_heatmap(
    cells: pd.DataFrame,
    output_path: str | Path,
    simulation_id: str | None = None,
    timepoint: int | None = None,
    bins: tuple[int, int] = (60, 30),
) -> None:
    if cells.empty or not {"x", "y", "local_signal"}.issubset(cells.columns):
        return
    snap = _choose_snapshot(cells, simulation_id, timepoint).copy()
    if snap.empty:
        return
    for col in ["x", "y", "local_signal"]:
        snap[col] = pd.to_numeric(snap[col], errors="coerce")
    snap = snap.dropna(subset=["x", "y", "local_signal"])
    if snap.empty:
        return
    wall_y = float(pd.to_numeric(cells["y"], errors="coerce").max())
    y_wall = np.maximum(0.0, wall_y - snap["y"].to_numpy())
    heat, xedges, yedges = np.histogram2d(
        snap["x"].to_numpy(),
        y_wall,
        bins=bins,
        weights=snap["local_signal"].to_numpy(),
    )
    counts, _, _ = np.histogram2d(snap["x"].to_numpy(), y_wall, bins=[xedges, yedges])
    mean_signal = np.divide(heat, counts, out=np.zeros_like(heat), where=counts > 0).T

    plt.figure(figsize=(8, 4))
    plt.imshow(
        mean_signal,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    plt.axhline(0, color="white", linewidth=1.0)
    plt.xlabel("x position along wall")
    plt.ylabel("distance from wall")
    plt.title("Local QS signal heatmap")
    plt.colorbar(label="mean local signal")
    save_fig(output_path)


def plot_flow_field(
    flow_field: pd.DataFrame,
    output_path: str | Path,
    simulation_id: str | None = None,
) -> None:
    if flow_field.empty or not {"x", "y", "velocity_x", "velocity_y", "shear"}.issubset(flow_field.columns):
        return
    flow = flow_field.copy()
    if simulation_id is None and "simulation_id" in flow.columns:
        simulation_id = str(flow["simulation_id"].dropna().astype(str).iloc[0])
    if simulation_id is not None and "simulation_id" in flow.columns:
        flow = flow[flow["simulation_id"].astype(str) == str(simulation_id)]
    if flow.empty:
        return
    for col in ["x", "y", "velocity_x", "velocity_y", "shear"]:
        flow[col] = pd.to_numeric(flow[col], errors="coerce")
    flow = flow.dropna(subset=["x", "y", "velocity_x", "velocity_y", "shear"])
    if flow.empty:
        return
    wall_y = float(pd.to_numeric(flow["y"], errors="coerce").max())
    flow["distance_from_wall"] = np.maximum(0.0, wall_y - flow["y"])

    plt.figure(figsize=(8, 4))
    sc = plt.scatter(
        flow["x"],
        flow["distance_from_wall"],
        c=flow["shear"],
        s=18,
        cmap="viridis",
        alpha=0.85,
    )
    step = max(1, len(flow) // 600)
    q = flow.iloc[::step]
    plt.quiver(
        q["x"],
        q["distance_from_wall"],
        q["velocity_x"],
        -q["velocity_y"],
        color="black",
        alpha=0.45,
        width=0.002,
        scale=25,
    )
    plt.axhline(0, color="black", linewidth=1.2, label="wall")
    plt.xlabel("x position along wall")
    plt.ylabel("distance from wall")
    plt.title("Flow and shear field")
    plt.colorbar(sc, label="shear")
    plt.gca().set_aspect("equal", adjustable="box")
    save_fig(output_path)
