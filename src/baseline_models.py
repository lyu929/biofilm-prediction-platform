from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .config import get_path, load_config
    from .dataset_builder import load_processed_dataset
    from .evaluate import regression_metrics
    from .visualize import plot_feature_importance, plot_pred_vs_true
except ImportError:
    from config import get_path, load_config
    from dataset_builder import load_processed_dataset
    from evaluate import regression_metrics
    from visualize import plot_feature_importance, plot_pred_vs_true


def _models(random_seed: int = 1):
    return {
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=random_seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=random_seed)),
            ]
        ),
    }


def train_baseline_models(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    target: str = "biofilm_thickness",
    config_path: str | Path | None = None,
    random_seed: int = 1,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    if X is None or y is None:
        _, X, y = load_processed_dataset(config_path=config_path, target=target)
    if len(X) < 3:
        raise ValueError("At least 3 samples are needed to train baseline models.")

    model_dir = get_path("models", cfg)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)

    test_size = min(0.3, max(1 / len(X), float(cfg.get("modeling", {}).get("test_size", 0.2))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    metrics_rows = []
    predictions = {}
    fitted = {}
    for name, model in _models(random_seed).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        metrics_rows.append({"model": name, "target": target, **metrics})
        predictions[name] = pred
        fitted[name] = model
        joblib.dump(model, model_dir / f"{target}_{name}.joblib")

        estimator = model.named_steps.get("model")
        if hasattr(estimator, "feature_importances_"):
            imp = pd.Series(estimator.feature_importances_, index=X.columns)
            imp.to_csv(result_dir / f"{target}_{name}_feature_importance.csv")
            plot_feature_importance(imp, fig_dir / f"{target}_{name}_feature_importance.png")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
    metrics_df.to_csv(result_dir / "baseline_metrics.csv", index=False)
    best_name = metrics_df.iloc[0]["model"]
    joblib.dump(fitted[best_name], model_dir / f"baseline_best_{target}.joblib")
    plot_pred_vs_true(y_test, predictions[best_name], "Best baseline: predicted vs true", fig_dir / "pred_vs_true_baseline.png")

    if len(X) < 10:
        warnings.warn("Very small dataset: baseline metrics are only a smoke-test signal.")
    return metrics_df


if __name__ == "__main__":
    print(train_baseline_models().to_string(index=False))
