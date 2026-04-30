from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd

try:
    from .config import TARGET_COLUMNS, get_path, load_config
    from .feature_engineering import build_feature_table
except ImportError:
    from config import TARGET_COLUMNS, get_path, load_config
    from feature_engineering import build_feature_table


PARAMETER_COLUMNS = [
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


def _load_template(model_dir: Path, target: str) -> dict[str, Any]:
    target_path = model_dir / f"feature_template_{target}.joblib"
    path = target_path if target_path.exists() else model_dir / "feature_template.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Feature template not found: {path}. Build a training dataset first.")
    return joblib.load(path)


def _align_features(features: pd.DataFrame, template: dict[str, Any]) -> pd.DataFrame:
    columns = template["feature_columns"]
    fill_values = template.get("fill_values", {})
    if features.empty:
        row = {col: fill_values.get(col, 0.0) for col in columns}
        return pd.DataFrame([row], columns=columns)
    row = {}
    first = features.iloc[0].to_dict()
    for col in columns:
        row[col] = first.get(col, fill_values.get(col, 0.0))
    out = pd.DataFrame([row], columns=columns)
    return out.apply(pd.to_numeric, errors="coerce").fillna(pd.Series(fill_values)).fillna(0.0)


def _load_model(model_dir: Path, target: str, model_type: str):
    if model_type == "baseline":
        path = model_dir / f"baseline_best_{target}.joblib"
        if not path.exists():
            candidates = sorted(model_dir.glob(f"{target}_*.joblib"))
            if not candidates:
                raise FileNotFoundError(f"No baseline model found for target '{target}'.")
            path = candidates[0]
        return joblib.load(path)
    if model_type == "mlp":
        import torch
        try:
            from .mlp_model import BiofilmMLP
        except ImportError:
            from mlp_model import BiofilmMLP

        model_path = model_dir / f"mlp_model_{target}.pt"
        scaler_path = model_dir / f"mlp_scaler_{target}.joblib"
        if not model_path.exists():
            model_path = model_dir / "mlp_model.pt"
        if not scaler_path.exists():
            scaler_path = model_dir / "mlp_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"MLP model/scaler not found for target '{target}'.")
        checkpoint = torch.load(model_path, map_location="cpu")
        model = BiofilmMLP(input_dim=int(checkpoint["input_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scaler = joblib.load(scaler_path)
        return model, scaler
    raise ValueError("model_type must be baseline or mlp")


def _predict_frame(features: pd.DataFrame, target: str, model_type: str, config_path: str | Path | None = None) -> float:
    cfg = load_config(config_path)
    model_dir = get_path("models", cfg)
    template = _load_template(model_dir, target)
    X = _align_features(features, template)

    if model_type == "baseline":
        model = _load_model(model_dir, target, model_type)
        return float(np.asarray(model.predict(X)).ravel()[0])

    import torch

    model, scaler = _load_model(model_dir, target, model_type)
    Xs = scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()[0]
    return float(pred)


def build_prediction_features(
    params: dict[str, Any],
    early_cells: pd.DataFrame | None = None,
    early_clusters: pd.DataFrame | None = None,
    flow_field: pd.DataFrame | None = None,
    early_timepoints: int = 10,
) -> pd.DataFrame:
    sid = str(params.get("simulation_id", "new_prediction"))
    sim_row = {"simulation_id": sid}
    for col in PARAMETER_COLUMNS:
        sim_row[col] = params.get(col, 0.0 if col != "random_seed" else 0)

    def prep(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        if "simulation_id" not in out.columns:
            out["simulation_id"] = sid
        return out

    data = {
        "simulations": pd.DataFrame([sim_row]),
        "cells": prep(early_cells),
        "clusters": prep(early_clusters),
        "flow_field": prep(flow_field),
    }
    return build_feature_table(data, early_timepoints=early_timepoints)


def predict_from_parameters(
    params: dict[str, Any],
    target: str = "biofilm_thickness",
    model_type: str = "baseline",
    config_path: str | Path | None = None,
) -> dict[str, float]:
    targets = TARGET_COLUMNS if target == "all" else [target]
    features = build_prediction_features(params)
    preds = {}
    for tgt in targets:
        try:
            preds[f"predicted_{tgt}"] = _predict_frame(features, tgt, model_type, config_path)
        except Exception as exc:
            warnings.warn(f"Skipping target '{tgt}': {exc}")
    return preds


def predict_from_early_data(
    params: dict[str, Any],
    early_cells: pd.DataFrame,
    early_clusters: pd.DataFrame | None = None,
    flow_field: pd.DataFrame | None = None,
    target: str = "biofilm_thickness",
    model_type: str = "baseline",
    early_timepoints: int = 10,
    config_path: str | Path | None = None,
) -> dict[str, float]:
    targets = TARGET_COLUMNS if target == "all" else [target]
    features = build_prediction_features(
        params,
        early_cells=early_cells,
        early_clusters=early_clusters,
        flow_field=flow_field,
        early_timepoints=early_timepoints,
    )
    preds = {}
    for tgt in targets:
        try:
            preds[f"predicted_{tgt}"] = _predict_frame(features, tgt, model_type, config_path)
        except Exception as exc:
            warnings.warn(f"Skipping target '{tgt}': {exc}")
    return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict final biofilm outcomes from parameters.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", default="biofilm_thickness")
    parser.add_argument("--model_type", choices=["baseline", "mlp"], default="baseline")
    for col in PARAMETER_COLUMNS:
        parser.add_argument(f"--{col}", type=float if col != "random_seed" and col != "runtime" else int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {col: getattr(args, col) for col in PARAMETER_COLUMNS if getattr(args, col) is not None}
    preds = predict_from_parameters(params, target=args.target, model_type=args.model_type, config_path=args.config)
    print(pd.DataFrame([preds]).to_string(index=False))


if __name__ == "__main__":
    main()
