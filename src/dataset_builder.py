from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
try:
    import joblib
except ImportError:  # pragma: no cover
    import pickle

    class _PickleJoblibFallback:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = _PickleJoblibFallback()

try:
    from .config import TARGET_COLUMNS, get_path, load_config
    from .feature_engineering import build_feature_table
except ImportError:
    from config import TARGET_COLUMNS, get_path, load_config
    from feature_engineering import build_feature_table


def build_training_dataset(
    data: dict[str, pd.DataFrame],
    target: str = "biofilm_thickness",
    early_timepoints: int = 10,
    config_path: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if target not in TARGET_COLUMNS:
        warnings.warn(f"Target '{target}' is not in the standard target list: {TARGET_COLUMNS}")

    features = build_feature_table(data, early_timepoints=early_timepoints)
    summary = data.get("simulation_summary", pd.DataFrame()).copy()
    if features.empty:
        raise ValueError("Feature table is empty. Load simulations/cells/clusters/flow_field first.")
    if summary.empty or "simulation_id" not in summary.columns or target not in summary.columns:
        raise ValueError(f"simulation_summary must contain simulation_id and target column '{target}'.")

    labels = summary[["simulation_id"] + [c for c in TARGET_COLUMNS if c in summary.columns]].copy()
    dataset = features.merge(labels, on="simulation_id", how="inner")
    if dataset.empty:
        raise ValueError("No matching simulation_id values between features and simulation_summary.")

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        dataset[col] = dataset[col].fillna(dataset[col].median() if not dataset[col].dropna().empty else 0.0)
    dataset = dataset.fillna(0.0)

    label_cols = [c for c in TARGET_COLUMNS if c in dataset.columns]
    feature_cols = [c for c in dataset.columns if c not in {"simulation_id"} | set(label_cols)]
    X = dataset[feature_cols].copy()
    y = dataset[target].copy()

    if save:
        cfg = load_config(config_path)
        processed = get_path("processed_data", cfg)
        model_dir = get_path("models", cfg)
        dataset.to_csv(processed / "training_dataset.csv", index=False)
        X.to_csv(processed / "X.csv", index=False)
        y.to_frame(name=target).to_csv(processed / "y.csv", index=False)
        template = {
            "target": target,
            "early_timepoints": early_timepoints,
            "feature_columns": feature_cols,
            "fill_values": X.median(numeric_only=True).fillna(0.0).to_dict(),
            "label_columns": label_cols,
        }
        joblib.dump(template, model_dir / "feature_template.joblib")
        joblib.dump(template, model_dir / f"feature_template_{target}.joblib")
    return dataset, X, y


def load_processed_dataset(
    config_path: str | Path | None = None,
    target: str = "biofilm_thickness",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cfg = load_config(config_path)
    processed = get_path("processed_data", cfg)
    dataset_path = processed / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    label_cols = [c for c in TARGET_COLUMNS if c in dataset.columns]
    feature_cols = [c for c in dataset.columns if c not in {"simulation_id"} | set(label_cols)]
    if target not in dataset.columns:
        raise ValueError(f"Target '{target}' not found in {dataset_path}")
    return dataset, dataset[feature_cols], dataset[target]


if __name__ == "__main__":
    from data_loader import load_all_data

    data = load_all_data("csv")
    ds, X, y = build_training_dataset(data)
    print(ds.shape, X.shape, y.shape)
