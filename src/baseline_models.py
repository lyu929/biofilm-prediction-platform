from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .config import get_path, load_config
    from .dataset_builder import load_processed_dataset
    from .evaluate import regression_metrics
    from .visualize import plot_feature_importance, plot_pred_vs_true, plot_residuals
except ImportError:
    from config import get_path, load_config
    from dataset_builder import load_processed_dataset
    from evaluate import regression_metrics
    from visualize import plot_feature_importance, plot_pred_vs_true, plot_residuals


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


# ---------- Hyperparameter search grids ----------
_PARAM_GRIDS = {
    "ridge_regression": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5],
    },
    "linear_regression": {},
}


def tune_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    target: str = "biofilm_thickness",
    n_splits: int = 5,
    random_seed: int = 1,
    config_path: str | Path | None = None,
) -> dict[str, Pipeline]:
    """GridSearchCV 超参数调优，返回每个模型调优后的最佳 pipeline。"""
    cfg = load_config(config_path)
    result_dir = get_path("results", cfg)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    best_models: dict[str, Pipeline] = {}
    rows = []

    for name, pipeline in _models(random_seed).items():
        grid = _PARAM_GRIDS.get(name, {})
        if grid:
            search = GridSearchCV(
                pipeline,
                grid,
                cv=kf,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
        else:
            pipeline.fit(X, y)
            best_pipe = pipeline
            best_params = {}

        best_models[name] = best_pipe
        rows.append({"model": name, "target": target, "best_params": str(best_params)})

    pd.DataFrame(rows).to_csv(result_dir / f"tuning_best_params_{target}.csv", index=False)
    return best_models


def cross_validate_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    target: str = "biofilm_thickness",
    n_splits: int = 5,
    random_seed: int = 1,
    config_path: str | Path | None = None,
    tune: bool = True,
) -> pd.DataFrame:
    """K-fold 交叉验证，返回每个模型在各折上的均值±标准差指标。"""
    cfg = load_config(config_path)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)

    n_splits = min(max(2, n_splits), len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    model_dict = _models(random_seed)

    rows = []
    for name, pipeline in model_dict.items():
        estimator = pipeline
        grid = _PARAM_GRIDS.get(name, {})
        if tune and grid and len(X) >= n_splits * 3:
            inner_splits = min(3, n_splits)
            estimator = GridSearchCV(
                pipeline,
                grid,
                cv=KFold(n_splits=inner_splits, shuffle=True, random_state=random_seed),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                refit=True,
            )
        cv_result = cross_validate(
            estimator, X, y,
            cv=kf,
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error",
                     "r2": "r2"},
            return_train_score=True,
            n_jobs=-1,
        )
        rows.append({
            "model": name,
            "target": target,
            "MAE_mean": -cv_result["test_mae"].mean(),
            "MAE_std": cv_result["test_mae"].std(),
            "RMSE_mean": -cv_result["test_rmse"].mean(),
            "RMSE_std": cv_result["test_rmse"].std(),
            "R2_mean": cv_result["test_r2"].mean(),
            "R2_std": cv_result["test_r2"].std(),
            "Train_R2_mean": cv_result["train_r2"].mean(),
        })

    cv_df = pd.DataFrame(rows).sort_values("RMSE_mean")
    cv_df.to_csv(result_dir / f"cv_results_{target}.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    lower_errs = np.minimum(cv_df["RMSE_mean"].values, cv_df["RMSE_std"].values)
    upper_errs = cv_df["RMSE_std"].values
    ax.bar(cv_df["model"], cv_df["RMSE_mean"],
           yerr=[lower_errs, upper_errs],
           capsize=5, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (mean ± std)")
    ax.set_title(f"{n_splits}-Fold CV RMSE — {target}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    ax.set_ylim(bottom=0)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"cv_rmse_{target}.png", dpi=180, bbox_inches="tight")
    plt.close()

    return cv_df


def train_baseline_models(
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    target: str = "biofilm_thickness",
    config_path: str | Path | None = None,
    random_seed: int = 1,
    run_cv: bool = False,
    tune: bool = False,
    permutation_repeats: int = 5,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    if X is None or y is None:
        _, X, y = load_processed_dataset(config_path=config_path, target=target)
    if len(X) < 3:
        raise ValueError("At least 3 samples are needed to train baseline models.")

    model_dir = get_path("models", cfg)
    result_dir = get_path("results", cfg)
    fig_dir = get_path("figures", cfg)

    cv_df = None
    if run_cv and len(X) >= 10:
        cv_df = cross_validate_baseline_models(
            X, y, target=target, random_seed=random_seed,
            config_path=config_path, tune=tune,
        )

    test_size = min(0.3, max(1 / len(X), float(cfg.get("modeling", {}).get("test_size", 0.2))))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    if tune and len(X) >= 15:
        final_models = tune_baseline_models(
            X_train, y_train, target=target, random_seed=random_seed,
            config_path=config_path,
        )
    else:
        final_models = _models(random_seed)
        for name, m in final_models.items():
            m.fit(X_train, y_train)

    metrics_rows = []
    predictions = {}
    for name, model in final_models.items():
        if not tune or len(X) < 15:
            pass  # already fitted above
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        metrics_rows.append({"model": name, "target": target, **metrics})
        predictions[name] = pred
        joblib.dump(model, model_dir / f"{target}_{name}.joblib")

        estimator = model.named_steps.get("model")
        if hasattr(estimator, "feature_importances_"):
            imp = pd.Series(estimator.feature_importances_, index=X.columns)
            imp.to_csv(result_dir / f"{target}_{name}_feature_importance.csv")
            plot_feature_importance(imp, fig_dir / f"{target}_{name}_feature_importance.png")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
    metrics_df.to_csv(result_dir / "baseline_metrics.csv", index=False)
    metrics_df.to_csv(result_dir / f"baseline_metrics_{target}.csv", index=False)
    best_name = metrics_df.iloc[0]["model"]
    joblib.dump(final_models[best_name], model_dir / f"baseline_best_{target}.joblib")
    plot_pred_vs_true(
        y_test, predictions[best_name],
        "Best baseline: predicted vs true",
        fig_dir / "pred_vs_true_baseline.png",
    )
    plot_pred_vs_true(
        y_test,
        predictions[best_name],
        f"Best baseline ({best_name}): predicted vs true",
        fig_dir / f"baseline_{target}_pred_vs_true.png",
    )
    plot_residuals(
        y_test,
        predictions[best_name],
        f"Best baseline ({best_name}): residuals",
        fig_dir / f"baseline_{target}_residuals.png",
    )

    if permutation_repeats > 0 and len(X_test) >= 5:
        try:
            perm = permutation_importance(
                final_models[best_name],
                X_test,
                y_test,
                n_repeats=permutation_repeats,
                random_state=random_seed,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            perm_series = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
            perm_series.to_csv(result_dir / f"{target}_{best_name}_permutation_importance.csv")
            plot_feature_importance(
                perm_series,
                fig_dir / f"{target}_{best_name}_permutation_importance.png",
            )
        except Exception as exc:
            warnings.warn(f"Permutation importance skipped for {target}: {exc}")

    if len(X) < 10:
        warnings.warn("Very small dataset: baseline metrics are only a smoke-test signal.")

    return metrics_df


if __name__ == "__main__":
    print(train_baseline_models().to_string(index=False))
