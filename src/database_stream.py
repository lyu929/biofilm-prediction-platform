from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

try:
    from .database import create_db_engine
    from .database_init import create_tables
except ImportError:
    from database import create_db_engine
    from database_init import create_tables


class DatabaseStreamWriter:
    """Append browser-simulation snapshots to the relational database.

    The browser simulation sends JSON records over the existing WebSocket used
    by the flow solver. This writer turns those JSON payloads into the standard
    tables visible in DBeaver.
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        self.config_path = Path(config_path)
        self.engine = None
        self.flow_written: set[str] = set()

    def _ensure_engine(self):
        if self.engine is None:
            create_tables(config_path=self.config_path)
            self.engine = create_db_engine(config_path=self.config_path)
        return self.engine

    @staticmethod
    def _clean_records(records: list[dict[str, Any]] | None) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].where(pd.notnull(df[col]), None)
        return df

    def _upsert_one(self, table: str, row: dict[str, Any], key: str = "simulation_id") -> None:
        engine = self._ensure_engine()
        if not row:
            return

        columns = list(row.keys())
        insert_cols = ", ".join(columns)
        values = ", ".join([f":{c}" for c in columns])
        update_cols = [c for c in columns if c != key]

        dialect = engine.dialect.name.lower()
        if dialect in {"mysql", "mariadb"}:
            updates = ", ".join([f"{c}=VALUES({c})" for c in update_cols])
            sql = f"INSERT INTO {table} ({insert_cols}) VALUES ({values}) ON DUPLICATE KEY UPDATE {updates}"
        elif dialect == "postgresql":
            updates = ", ".join([f"{c}=EXCLUDED.{c}" for c in update_cols])
            sql = (
                f"INSERT INTO {table} ({insert_cols}) VALUES ({values}) "
                f"ON CONFLICT ({key}) DO UPDATE SET {updates}"
            )
        else:
            raise ValueError(f"Unsupported database dialect for upsert: {dialect}")

        with engine.begin() as conn:
            conn.execute(text(sql), row)

    def write_simulation_start(self, simulation: dict[str, Any]) -> None:
        self._upsert_one("simulations", simulation)

    def write_snapshot(self, payload: dict[str, Any]) -> dict[str, int]:
        engine = self._ensure_engine()
        simulation_id = str(payload.get("simulation_id", ""))
        counts = {"cells": 0, "clusters": 0, "flow_field": 0, "simulation_summary": 0}

        cells = self._clean_records(payload.get("cells"))
        if not cells.empty:
            cells.to_sql("cells", engine, if_exists="append", index=False, method="multi", chunksize=5000)
            counts["cells"] = len(cells)

        clusters = self._clean_records(payload.get("clusters"))
        if not clusters.empty:
            clusters.to_sql("clusters", engine, if_exists="append", index=False, method="multi", chunksize=5000)
            counts["clusters"] = len(clusters)

        flow = self._clean_records(payload.get("flow_field"))
        if simulation_id and simulation_id not in self.flow_written and not flow.empty:
            flow.to_sql("flow_field", engine, if_exists="append", index=False, method="multi", chunksize=5000)
            self.flow_written.add(simulation_id)
            counts["flow_field"] = len(flow)

        summary = payload.get("simulation_summary") or {}
        if summary:
            self._upsert_one("simulation_summary", summary)
            counts["simulation_summary"] = 1

        return counts
