from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_models import cross_validate_baseline_models, train_baseline_models
from src.config import TARGET_COLUMNS, get_path, load_config
from src.data_loader import load_all_data, load_from_excel
from src.database import read_table, test_connection
from src.database_init import create_tables, reset_tables, show_table_counts
from src.database_writer import write_all_results
from src.dataset_builder import build_training_dataset
from src.eda import run_eda
from src.feature_engineering import aggregate_cells_by_timepoint
from src.predict import PARAMETER_COLUMNS, predict_from_early_data, predict_from_parameters
from src.ablation import run_ablation, run_ablation_all_targets


st.set_page_config(page_title="Biofilm Prediction Platform", layout="wide")


@st.cache_data(show_spinner=False)
def cached_config():
    return load_config(ROOT / "config.yaml")


def db_form_config() -> dict:
    cfg = cached_config().get("database", {})
    with st.form("db_form"):
        db_type = st.selectbox("Database type", ["mysql", "postgresql"], index=0 if cfg.get("type", "mysql") == "mysql" else 1)
        host = st.text_input("Host", cfg.get("host", "localhost"))
        port_default = int(cfg.get("port", 3306 if db_type == "mysql" else 5432))
        port = st.number_input("Port", min_value=1, max_value=65535, value=port_default)
        user = st.text_input("User", cfg.get("user", "root" if db_type == "mysql" else "postgres"))
        password = st.text_input("Password", cfg.get("password", ""), type="password")
        database = st.text_input("Database", cfg.get("database", "biofilm_db"))
        submitted = st.form_submit_button("Use Connection")
    db = {"type": db_type, "host": host, "port": int(port), "user": user, "password": password, "database": database}
    st.session_state["db_config"] = db
    return db


def current_db_config() -> dict:
    if "db_config" not in st.session_state:
        st.session_state["db_config"] = cached_config().get("database", {})
    return st.session_state["db_config"]


def read_uploaded_csv(label: str):
    file = st.file_uploader(label, type=["csv"], key=label)
    if file is None:
        return pd.DataFrame()
    return pd.read_csv(file)


def show_data_preview(data: dict[str, pd.DataFrame]) -> None:
    for name, df in data.items():
        with st.expander(f"{name} ({len(df):,} rows)", expanded=False):
            st.dataframe(df.head(200), use_container_width=True)


def training_xy_for_target(target: str) -> tuple[pd.DataFrame | None, pd.Series | None]:
    """Return the saved feature matrix and target-aligned y from the built dataset."""
    X = st.session_state.get("train_X")
    ds = st.session_state.get("train_dataset")
    if X is None:
        return None, None
    if ds is not None and target in ds.columns:
        return X, ds[target]
    y = st.session_state.get("train_y")
    trained_target = st.session_state.get("trained_target")
    if y is not None and trained_target == target:
        return X, y
    return X, None


def page_intro() -> None:
    st.title("Biofilm Simulation Prediction Platform", anchor=False)
    st.write(
        "A research workflow for predicting final biofilm properties from early-stage simulation observations. "
        "Runs parameter sweep batches, stores results in a relational database, engineers early-timepoint features, "
        "and trains predictive models to forecast biofilm outcomes without completing the full simulation."
    )
    st.subheader("Prediction Targets", anchor=False)
    st.write(", ".join(TARGET_COLUMNS))
    st.subheader("Data Flow", anchor=False)
    st.write(
        "Simulation parameters and cell- and cluster-level time-series data are persisted across five standard tables "
        "(simulations, cells, clusters, flow_field, simulation_summary). Each `simulation_id` is aggregated into one "
        "training sample using input parameters, early cell/cluster dynamics, and flow-field statistics."
    )


def page_database() -> None:
    st.title("Database Connection")
    db = db_form_config()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Test Connection"):
            try:
                test_connection(db_config=db)
                st.success("Connection OK")
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Create Tables"):
            try:
                create_tables(db_config=db)
                st.success("Tables ready")
            except Exception as exc:
                st.error(str(exc))
    with c3:
        if st.button("Reset Tables"):
            try:
                reset_tables(db_config=db)
                st.warning("Tables reset")
            except Exception as exc:
                st.error(str(exc))
    with c4:
        if st.button("Show Counts"):
            try:
                st.session_state["table_counts"] = show_table_counts(db_config=db)
            except Exception as exc:
                st.error(str(exc))

    if "table_counts" in st.session_state:
        st.dataframe(st.session_state["table_counts"], use_container_width=True)

    table = st.selectbox("Preview table", ["simulations", "cells", "clusters", "flow_field", "simulation_summary"])
    limit = st.number_input("Rows", min_value=10, max_value=1000, value=100, step=10)
    if st.button("Load Preview"):
        try:
            st.dataframe(read_table(table, db_config=db, limit=int(limit)), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def page_import() -> None:
    st.title("Simulation Data Import")
    source = st.radio("Import type", ["CSV files", "Excel workbook"], horizontal=True)
    data = {}
    if source == "CSV files":
        for table in ["simulations", "cells", "clusters", "flow_field", "simulation_summary"]:
            data[table] = read_uploaded_csv(f"Upload {table}.csv")
    else:
        excel = st.file_uploader("Upload Excel workbook", type=["xlsx", "xls"])
        if excel is not None:
            tmp = ROOT / "data/raw/uploaded_workbook.xlsx"
            tmp.write_bytes(excel.getvalue())
            data = load_from_excel(tmp)

    if data:
        show_data_preview(data)
        if st.button("Write Uploaded Data To Database"):
            try:
                write_all_results(data, db_config=current_db_config(), on_existing="replace")
                st.success("Uploaded data written to database")
            except Exception as exc:
                st.error(str(exc))


def page_analysis() -> None:
    st.title("Data Analysis")
    source = st.radio("Data source", ["csv", "database"], horizontal=True, key="analysis_source")
    target = st.selectbox("Target", TARGET_COLUMNS, key="analysis_target")
    if st.button("Load and Analyze"):
        try:
            data = load_all_data(source=source, db_config=current_db_config())
            ds, _, _ = build_training_dataset(data, target=target, save=False)
            run_eda(data, ds, target=target)
            st.session_state["analysis_data"] = data
            st.session_state["analysis_dataset"] = ds
            st.session_state["analysis_result_target"] = target
            st.success("EDA complete")
        except Exception as exc:
            st.error(str(exc))

    if "analysis_dataset" in st.session_state:
        ds = st.session_state["analysis_dataset"]
        st.subheader("Summary Statistics")
        st.dataframe(ds.describe().T, use_container_width=True)
        fig_dir = get_path("figures", cached_config())
        shown_target = st.session_state.get("analysis_result_target", target)
        for name in [
            f"{shown_target}_distribution.png",
            "correlation_heatmap.png",
            "attached_ratio_over_time.png",
            "qs_active_ratio_over_time.png",
            "eps_producing_ratio_over_time.png",
            "biofilm_cell_state_snapshot.png",
            "flow_shear_field.png",
        ]:
            path = fig_dir / name
            if path.exists():
                st.image(str(path), use_container_width=True)


def page_training() -> None:
    st.title("Model Training")
    source = st.radio("Data source", ["csv", "database"], horizontal=True, key="train_source")
    target = st.selectbox("Prediction target", TARGET_COLUMNS, key="train_target")
    early = st.number_input("Early timepoints", min_value=1, max_value=100, value=10)
    if st.button("Build Training Dataset"):
        try:
            data = load_all_data(source=source, db_config=current_db_config())
            ds, X, y = build_training_dataset(data, target=target, early_timepoints=int(early), save=True)
            st.session_state["train_dataset"] = ds
            st.session_state["train_X"] = X
            st.session_state["train_y"] = y
            st.session_state["trained_target"] = target
            st.success(f"Dataset built: {ds.shape[0]} samples, {X.shape[1]} features")
            st.dataframe(ds.head(50), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train Baseline Models"):
            try:
                X, y = training_xy_for_target(target)
                if X is None or y is None:
                    st.warning("Build the training dataset for this target first.")
                    return
                metrics = train_baseline_models(X, y, target=target, run_cv=False, tune=False)
                st.dataframe(metrics, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Train MLP Model"):
            try:
                from src.train_mlp import train_mlp_model
                X, y = training_xy_for_target(target)
                if X is None or y is None:
                    st.warning("Build the training dataset for this target first.")
                    return
                history = train_mlp_model(X, y, target=target)
                st.dataframe(history.tail(20), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    metrics_path = ROOT / "outputs/results/baseline_metrics.csv"
    if metrics_path.exists():
        st.subheader("Baseline Metrics")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)


def page_cross_validation() -> None:
    st.title("Cross-Validation & Hyperparameter Tuning", anchor=False)
    st.write(
        "Evaluate model generalisation with K-Fold cross-validation and automatically "
        "search for optimal hyperparameters using GridSearchCV."
    )

    target = st.selectbox("Prediction target", TARGET_COLUMNS, key="cv_target")
    n_splits = st.slider("K-Fold splits", min_value=3, max_value=10, value=5)
    tune = st.checkbox("Run hyperparameter tuning (GridSearchCV)", value=True)

    if st.button("Run Cross-Validation"):
        X, y = training_xy_for_target(target)
        if X is None or y is None:
            st.warning("Please build the training dataset first on the Model Training page.")
            return
        try:
            with st.spinner("Running cross-validation — this may take a few minutes..."):
                cv_df = cross_validate_baseline_models(
                    X, y, target=target, n_splits=n_splits, tune=tune
                )
            st.session_state["cv_df"] = cv_df
            st.success("Cross-validation complete!")
        except Exception as exc:
            st.error(str(exc))

    if "cv_df" in st.session_state:
        cv_df = st.session_state["cv_df"]
        st.subheader("CV Results (mean ± std)", anchor=False)
        display_cols = ["model", "target", "MAE_mean", "MAE_std",
                        "RMSE_mean", "RMSE_std", "R2_mean", "R2_std", "Train_R2_mean"]
        st.dataframe(
            cv_df[[c for c in display_cols if c in cv_df.columns]].style.format({
                "MAE_mean": "{:.4f}", "MAE_std": "{:.4f}",
                "RMSE_mean": "{:.4f}", "RMSE_std": "{:.4f}",
                "R2_mean": "{:.4f}", "R2_std": "{:.4f}",
                "Train_R2_mean": "{:.4f}",
            }),
            use_container_width=True,
        )

        fig_dir = get_path("figures", cached_config())
        cv_img = fig_dir / f"cv_rmse_{target}.png"
        if cv_img.exists():
            st.image(str(cv_img), use_container_width=True)

    result_dir = get_path("results", cached_config())
    params_path = result_dir / f"tuning_best_params_{target}.csv"
    if params_path.exists():
        st.subheader("Best Hyperparameters Found", anchor=False)
        import ast
        raw_df = pd.read_csv(params_path)
        expanded = []
        for _, row in raw_df.iterrows():
            try:
                params = ast.literal_eval(row["best_params"])
            except Exception:
                params = {}
            if not params:
                expanded.append({"Model": row["model"], "Parameter": "—", "Value": "—"})
            else:
                for k, v in params.items():
                    expanded.append({
                        "Model": row["model"],
                        "Parameter": k.replace("model__", ""),
                        "Value": v,
                    })
        st.dataframe(pd.DataFrame(expanded), use_container_width=True)


def page_ablation() -> None:
    st.title("Ablation Study", anchor=False)
    st.write(
        "Incrementally add feature groups to quantify each group's contribution to predictive performance.\n\n"
        "| Feature group | Contents |\n"
        "|---------------|----------|\n"
        "| `params_only` | 10 simulation parameters only |\n"
        "| `params_flow` | Parameters + flow-field features |\n"
        "| `params_cells` | Parameters + early cell time-series features |\n"
        "| `params_clusters` | Parameters + early cluster time-series features |\n"
        "| `all_features` | All features combined (default) |"
    )

    target = st.selectbox("Prediction target", ["all"] + TARGET_COLUMNS, key="ablation_target")
    n_splits = st.slider("K-Fold splits", min_value=3, max_value=10, value=5, key="ablation_splits")

    if st.button("Run Ablation Study"):
        X = st.session_state.get("train_X")
        _, y = training_xy_for_target(target) if target != "all" else (X, None)
        if X is None:
            st.warning("Please build the training dataset first on the Model Training page.")
            return
        try:
            with st.spinner("Running ablation study..."):
                if target == "all":
                    ablation_df = run_ablation_all_targets(X=X, n_splits=n_splits)
                else:
                    if y is None:
                        st.warning("Build the training dataset for this target first.")
                        return
                    ablation_df = run_ablation(
                        X=X, y=y, target=target, n_splits=n_splits
                    )
            st.session_state["ablation_df"] = ablation_df
            st.session_state["ablation_result_target"] = target
            st.success("Ablation study complete!")
        except Exception as exc:
            st.error(str(exc))

    if "ablation_df" in st.session_state:
        ablation_df = st.session_state["ablation_df"]
        st.subheader("Ablation Results", anchor=False)
        st.dataframe(
            ablation_df.style.format({
                "MAE_mean": "{:.4f}", "MAE_std": "{:.4f}",
                "RMSE_mean": "{:.4f}", "RMSE_std": "{:.4f}",
                "R2_mean": "{:.4f}", "R2_std": "{:.4f}",
            }),
            use_container_width=True,
        )

        fig_dir = get_path("figures", cached_config())
        t = st.session_state.get("ablation_result_target", target)
        if t == "all":
            for tgt in TARGET_COLUMNS:
                img = fig_dir / f"ablation_{tgt}.png"
                if img.exists():
                    st.image(str(img), caption=tgt, use_container_width=True)
        else:
            img = fig_dir / f"ablation_{t}.png"
            if img.exists():
                st.image(str(img), use_container_width=True)


def page_uncertainty() -> None:
    st.title("Prediction Uncertainty (Bootstrap CI)", anchor=False)
    st.write(
        "Estimate 95% confidence intervals for MAE, RMSE, and R² via bootstrap resampling, "
        "quantifying the statistical reliability of model evaluation results."
    )

    target = st.selectbox("Prediction target", TARGET_COLUMNS, key="ci_target")
    model_type = st.radio("Model", ["baseline", "mlp"], horizontal=True, key="ci_model")
    n_bootstrap = st.slider("Bootstrap iterations", 200, 2000, 1000, step=100)

    if st.button("Compute Bootstrap CI"):
        X, y = training_xy_for_target(target)
        if X is None or y is None:
            st.warning("Please build the training dataset first on the Model Training page.")
            return
        try:
            from sklearn.model_selection import train_test_split
            from src.evaluate import regression_metrics_with_ci

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if model_type == "baseline":
                import joblib
                model_dir = get_path("models", cached_config())
                model_path = model_dir / f"baseline_best_{target}.joblib"
                if not model_path.exists():
                    st.warning(f"No trained baseline model found ({model_path.name}). Please train first.")
                    return
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
            else:
                import torch
                import joblib
                from src.mlp_model import BiofilmMLP, LegacyBiofilmMLP
                model_dir = get_path("models", cached_config())
                ckpt_path = model_dir / f"mlp_model_{target}.pt"
                scaler_path = model_dir / f"mlp_scaler_{target}.joblib"
                if not ckpt_path.exists():
                    st.warning(f"No trained MLP model found ({ckpt_path.name}). Please train first.")
                    return
                ckpt = torch.load(ckpt_path, map_location="cpu")
                scaler = joblib.load(scaler_path)
                mlp = BiofilmMLP(
                    input_dim=ckpt["input_dim"],
                    hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64, 32))),
                    dropout=ckpt.get("dropout", 0.3),
                )
                try:
                    mlp.load_state_dict(ckpt["state_dict"])
                except RuntimeError:
                    mlp = LegacyBiofilmMLP(input_dim=ckpt["input_dim"])
                    mlp.load_state_dict(ckpt["state_dict"])
                mlp.eval()
                X_test_s = scaler.transform(X_test)
                with torch.no_grad():
                    y_pred = mlp(torch.tensor(X_test_s, dtype=torch.float32)).numpy().ravel()

            with st.spinner(f"Bootstrap resampling ({n_bootstrap} iterations)..."):
                metrics = regression_metrics_with_ci(
                    y_test, y_pred, n_bootstrap=n_bootstrap
                )
            st.session_state["ci_metrics"] = metrics

        except Exception as exc:
            st.error(str(exc))

    if "ci_metrics" in st.session_state:
        m = st.session_state["ci_metrics"]
        rows = []
        for metric in ("MAE", "RMSE", "R2"):
            rows.append({
                "Metric": metric,
                "Point Estimate": f"{m[metric]:.4f}",
                "95% CI Lower": f"{m[f'{metric}_CI_lower']:.4f}",
                "95% CI Upper": f"{m[f'{metric}_CI_upper']:.4f}",
            })
        st.subheader("Bootstrap 95% Confidence Intervals", anchor=False)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def parameter_inputs(prefix: str = "") -> dict:
    defaults = {
        "flow_rate": 0.5,
        "adhesion_wall": 0.6,
        "adhesion_cell": 0.5,
        "diffusion_rate": 0.05,
        "signal_decay": 0.01,
        "qs_threshold": 0.6,
        "division_rate": 0.01,
        "eps_rate": 0.02,
        "runtime": 300,
        "random_seed": 1,
    }
    cols = st.columns(2)
    params = {}
    for i, name in enumerate(PARAMETER_COLUMNS):
        with cols[i % 2]:
            if name in {"runtime", "random_seed"}:
                params[name] = st.number_input(name, value=int(defaults[name]), key=f"{prefix}{name}")
            else:
                params[name] = st.number_input(name, value=float(defaults[name]), key=f"{prefix}{name}", format="%.5f")
    return params


def page_parameter_prediction() -> None:
    st.title("New Parameter Prediction")
    target = st.selectbox("Target", ["all"] + TARGET_COLUMNS, key="param_pred_target")
    model_type = st.radio("Model", ["baseline", "mlp"], horizontal=True, key="param_model")
    params = parameter_inputs("param_")
    if st.button("Predict"):
        try:
            preds = predict_from_parameters(params, target=target, model_type=model_type)
            st.dataframe(pd.DataFrame([preds]), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def page_early_prediction() -> None:
    st.title("Early Data Prediction")
    target = st.selectbox("Target", ["all"] + TARGET_COLUMNS, key="early_pred_target")
    model_type = st.radio("Model", ["baseline", "mlp"], horizontal=True, key="early_model")
    early = st.number_input("Early timepoints", min_value=1, max_value=100, value=10, key="early_pred_count")
    params = parameter_inputs("early_")
    cells = read_uploaded_csv("Upload early cells CSV")
    clusters = read_uploaded_csv("Upload early clusters CSV")
    flow = read_uploaded_csv("Upload optional flow_field CSV")
    if st.button("Predict From Early Data"):
        try:
            preds = predict_from_early_data(
                params,
                early_cells=cells,
                early_clusters=clusters,
                flow_field=flow,
                target=target,
                model_type=model_type,
                early_timepoints=int(early),
            )
            st.dataframe(pd.DataFrame([preds]), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def page_visualization() -> None:
    st.title("Model Result Visualization")
    fig_dir = ROOT / "outputs/figures"
    existing = sorted(fig_dir.glob("*.png"))
    if not existing:
        st.info("No figures generated yet.")
        return
    selected = st.multiselect("Figures", [p.name for p in existing], default=[p.name for p in existing[:4]])
    for name in selected:
        st.image(str(fig_dir / name), caption=name, use_container_width=True)


PAGES = {
    "Project Intro": page_intro,
    "Database Connection": page_database,
    "Simulation Data Import": page_import,
    "Data Analysis": page_analysis,
    "Model Training": page_training,
    "Cross-Validation & Tuning": page_cross_validation,
    "Ablation Study": page_ablation,
    "Prediction Uncertainty": page_uncertainty,
    "New Parameter Prediction": page_parameter_prediction,
    "Early Data Prediction": page_early_prediction,
    "Model Visualization": page_visualization,
}


def main() -> None:
    st.sidebar.title("Biofilm Platform")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()
