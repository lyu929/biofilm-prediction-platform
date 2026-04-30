from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from sqlalchemy import inspect, text

try:
    from .config import TABLE_NAMES
    from .database import create_db_engine
    from .database_init import create_tables
except ImportError:
    from config import TABLE_NAMES
    from database import create_db_engine
    from database_init import create_tables


TABLE_BY_KEY = {
    "simulations": "simulations",
    "cells": "cells",
    "clusters": "clusters",
    "flow_field": "flow_field",
    "simulation_summary": "simulation_summary",
}


def _clean_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].where(pd.notnull(out[col]), None)
    return out


def _simulation_ids(df: pd.DataFrame) -> list[str]:
    if df.empty or "simulation_id" not in df.columns:
        return []
    return [str(v) for v in df["simulation_id"].dropna().unique().tolist()]


def _delete_existing(engine, simulation_ids: list[str]) -> None:
    if not simulation_ids:
        return
    placeholders = ", ".join([f":sid{i}" for i in range(len(simulation_ids))])
    params = {f"sid{i}": sid for i, sid in enumerate(simulation_ids)}
    with engine.begin() as conn:
        for table in ["cells", "clusters", "flow_field", "simulation_summary", "simulations"]:
            conn.execute(text(f"DELETE FROM {table} WHERE simulation_id IN ({placeholders})"), params)


def _existing_ids(engine, simulation_ids: list[str]) -> set[str]:
    if not simulation_ids:
        return set()
    placeholders = ", ".join([f":sid{i}" for i in range(len(simulation_ids))])
    params = {f"sid{i}": sid for i, sid in enumerate(simulation_ids)}
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT simulation_id FROM simulations WHERE simulation_id IN ({placeholders})"),
            params,
        ).fetchall()
    return {str(r[0]) for r in rows}


def _write_df(
    df: pd.DataFrame,
    table_name: str,
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
    mode: str = "append",
) -> None:
    if df.empty:
        warnings.warn(f"No rows to write for table '{table_name}'.")
        return
    create_tables(config_path=config_path, db_config=db_config)
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    if not inspect(engine).has_table(table_name):
        create_tables(config_path=config_path, db_config=db_config)
    df.to_sql(table_name, engine, if_exists=mode, index=False, method="multi", chunksize=5000)


def write_simulation_params(df: pd.DataFrame, **kwargs) -> None:
    _write_df(_clean_df(df), "simulations", **kwargs)


def write_cells(df: pd.DataFrame, **kwargs) -> None:
    _write_df(_clean_df(df), "cells", **kwargs)


def write_clusters(df: pd.DataFrame, **kwargs) -> None:
    _write_df(_clean_df(df), "clusters", **kwargs)


def write_flow_field(df: pd.DataFrame, **kwargs) -> None:
    _write_df(_clean_df(df), "flow_field", **kwargs)


def write_simulation_summary(df: pd.DataFrame, **kwargs) -> None:
    _write_df(_clean_df(df), "simulation_summary", **kwargs)


def write_all_results(
    results: dict[str, pd.DataFrame],
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
    on_existing: str = "replace",
) -> None:
    """Write all standard result tables.

    on_existing:
      - replace: delete previous rows for the same simulation_id, then append.
      - skip: skip every incoming simulation_id that already exists.
      - append: append rows without checking duplicates.
    """
    create_tables(config_path=config_path, db_config=db_config)
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    sim_df = _clean_df(results.get("simulations"))
    ids = _simulation_ids(sim_df)

    tables = {name: _clean_df(results.get(name)) for name in TABLE_NAMES}

    if on_existing == "replace":
        _delete_existing(engine, ids)
    elif on_existing == "skip":
        existing = _existing_ids(engine, ids)
        if existing:
            for name, df in tables.items():
                if "simulation_id" in df.columns:
                    tables[name] = df[~df["simulation_id"].astype(str).isin(existing)]
    elif on_existing != "append":
        raise ValueError("on_existing must be one of: replace, skip, append")

    for name in TABLE_NAMES:
        df = tables.get(name, pd.DataFrame())
        if not df.empty:
            df.to_sql(name, engine, if_exists="append", index=False, method="multi", chunksize=5000)
