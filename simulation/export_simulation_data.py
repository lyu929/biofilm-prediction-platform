from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd


STANDARD_TABLES = [
    "simulations",
    "cells",
    "clusters",
    "flow_field",
    "simulation_summary",
]


def empty_results() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame() for name in STANDARD_TABLES}


def append_or_create_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        warnings.warn(f"Skipping empty CSV export: {path}")
        return
    if path.exists() and path.stat().st_size > 0:
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def save_results_to_csv(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path = "data/raw",
    append: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for table in STANDARD_TABLES:
        df = results.get(table, pd.DataFrame())
        if df.empty:
            continue
        path = out / f"{table}.csv"
        if append:
            append_or_create_csv(path, df)
        else:
            df.to_csv(path, index=False)


def combine_results(result_list: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    combined = empty_results()
    for table in STANDARD_TABLES:
        frames = [r.get(table, pd.DataFrame()) for r in result_list if not r.get(table, pd.DataFrame()).empty]
        combined[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined


def write_results_to_database(
    results: dict[str, pd.DataFrame],
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
    on_existing: str = "replace",
) -> None:
    try:
        from src.database_writer import write_all_results
    except ImportError:
        from database_writer import write_all_results

    write_all_results(results, config_path=config_path, db_config=db_config, on_existing=on_existing)
