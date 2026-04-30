from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_models import train_baseline_models
from src.config import TARGET_COLUMNS, get_path, load_config
from src.data_loader import load_all_data, load_from_excel
from src.database import read_table, test_connection
from src.database_init import create_tables, reset_tables, show_table_counts
from src.database_writer import write_all_results
from src.dataset_builder import build_training_dataset
from src.eda import run_eda
from src.feature_engineering import aggregate_cells_by_timepoint
from src.predict import PARAMETER_COLUMNS, predict_from_early_data, predict_from_parameters


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


def page_intro() -> None:
    st.title("Biofilm Simulation Prediction Platform")
    st.write(
        "A research workflow for running biofilm simulation batches, storing results in a relational database, "
        "engineering early-timepoint features, training predictive models, and exploring predictions interactively."
    )
    st.subheader("Prediction Targets")
    st.write(", ".join(TARGET_COLUMNS))
    st.subheader("Data Flow")
    st.write(
        "Simulation parameters and process data are exported to standard tables. Each `simulation_id` is converted "
        "into one training sample using input parameters, early cells/clusters, and flow-field features."
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
            st.success("EDA complete")
        except Exception as exc:
            st.error(str(exc))

    if "analysis_dataset" in st.session_state:
        ds = st.session_state["analysis_dataset"]
        st.subheader("Summary Statistics")
        st.dataframe(ds.describe().T, use_container_width=True)
        fig_dir = get_path("figures", cached_config())
        for name in [f"{target}_distribution.png", "correlation_heatmap.png", "attached_ratio_over_time.png"]:
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
            st.session_state["train_X"] = X
            st.session_state["train_y"] = y
            st.session_state["train_target"] = target
            st.success(f"Dataset built: {ds.shape[0]} samples, {X.shape[1]} features")
            st.dataframe(ds.head(50), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train Baseline Models"):
            try:
                X = st.session_state.get("train_X")
                y = st.session_state.get("train_y")
                metrics = train_baseline_models(X, y, target=target)
                st.dataframe(metrics, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
    with c2:
        if st.button("Train MLP Model"):
            try:
                from src.train_mlp import train_mlp_model

                X = st.session_state.get("train_X")
                y = st.session_state.get("train_y")
                history = train_mlp_model(X, y, target=target)
                st.dataframe(history.tail(20), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    metrics_path = ROOT / "outputs/results/baseline_metrics.csv"
    if metrics_path.exists():
        st.subheader("Baseline Metrics")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)


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
    images = [
        "pred_vs_true_baseline.png",
        "mlp_loss_curve.png",
        "correlation_heatmap.png",
        "random_forest_feature_importance.png",
    ]
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
