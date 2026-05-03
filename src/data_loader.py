from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

try:
    from .config import PROJECT_ROOT, TABLE_NAMES, load_config
except ImportError:
    from config import PROJECT_ROOT, TABLE_NAMES, load_config


DEFAULT_CSV_PATHS = {
    "simulations": "data/raw/simulations.csv",
    "cells": "data/raw/cells.csv",
    "clusters": "data/raw/clusters.csv",
    "flow_field": "data/raw/flow_field.csv",
    "simulation_summary": "data/raw/simulation_summary.csv",
}


def _empty_data() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame() for name in TABLE_NAMES}


def load_from_database(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    try:
        from .database import read_table
    except ImportError:
        from database import read_table

    data = {}
    for table in TABLE_NAMES:
        try:
            data[table] = read_table(table, config_path=config_path, db_config=db_config)
        except Exception as exc:
            warnings.warn(f"Could not load table '{table}' from database: {exc}")
            data[table] = pd.DataFrame()
    return data


def load_from_csv(
    base_dir: str | Path | None = None,
    paths: dict[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    data = _empty_data()
    base = Path(base_dir) if base_dir else PROJECT_ROOT
    paths = paths or DEFAULT_CSV_PATHS
    for table in TABLE_NAMES:
        path = Path(paths.get(table, DEFAULT_CSV_PATHS[table]))
        if not path.is_absolute():
            path = base / path
        if path.exists():
            data[table] = pd.read_csv(path)
        else:
            warnings.warn(f"CSV file not found for '{table}': {path}")
    return data


def load_from_excel(excel_path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = _empty_data()
    xls = pd.ExcelFile(path)
    for table in TABLE_NAMES:
        if table in xls.sheet_names:
            data[table] = pd.read_excel(path, sheet_name=table)
        else:
            warnings.warn(f"Excel sheet '{table}' not found in {path}")
    return data


def load_all_data(
    source: str = "csv",
    config_path: str | Path | None = None,
    excel_path: str | Path | None = None,
    csv_base_dir: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    source = source.lower()
    if source == "database":
        return load_from_database(config_path=config_path, db_config=db_config)
    if source == "csv":
        if csv_base_dir is None and config_path is not None:
            cfg = load_config(config_path)
            csv_base_dir = PROJECT_ROOT
            raw_rel = cfg.get("paths", {}).get("raw_data")
            if raw_rel:
                paths = {table: Path(raw_rel) / f"{table}.csv" for table in TABLE_NAMES}
                return load_from_csv(base_dir=csv_base_dir, paths=paths)
        return load_from_csv(base_dir=csv_base_dir)
    if source == "excel":
        if not excel_path:
            cfg = load_config(config_path)
            excel_path = cfg.get("paths", {}).get("excel_file", PROJECT_ROOT / "data/raw/biofilm_data.xlsx")
        return load_from_excel(excel_path)
    raise ValueError("source must be one of: database, csv, excel")
