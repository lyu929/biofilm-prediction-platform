from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import load_config
except ImportError:
    from config import load_config


def _database_url(db_cfg: dict[str, Any], include_database: bool = True) -> str:
    db_type = (db_cfg.get("type") or "mysql").lower()
    host = db_cfg.get("host", "localhost")
    user = quote_plus(str(db_cfg.get("user", "")))
    password = quote_plus(str(db_cfg.get("password", "")))
    database = db_cfg.get("database", "")

    if db_type in {"mysql", "mariadb"}:
        port = int(db_cfg.get("port", 3306))
        db_part = f"/{database}" if include_database and database else ""
        return f"mysql+pymysql://{user}:{password}@{host}:{port}{db_part}?charset=utf8mb4"

    if db_type in {"postgres", "postgresql"}:
        port = int(db_cfg.get("port", 5432))
        db_name = database if include_database and database else db_cfg.get("maintenance_database", "postgres")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

    raise ValueError(f"Unsupported database type: {db_type}. Use mysql or postgresql.")


def create_db_engine(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
    include_database: bool = True,
) -> Engine:
    cfg = load_config(config_path) if db_config is None else {"database": db_config}
    url = _database_url(cfg.get("database", {}), include_database=include_database)
    return create_engine(url, pool_pre_ping=True, future=True)


def test_connection(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> bool:
    try:
        engine = create_db_engine(config_path=config_path, db_config=db_config)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        raise ConnectionError(
            "Database connection failed. Check config.yaml host, port, user, password, "
            f"database name, and driver installation. Original error: {exc}"
        ) from exc


def read_table(
    table_name: str,
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    sql = f"SELECT * FROM {table_name}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    try:
        return pd.read_sql_query(sql, engine)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to read table '{table_name}': {exc}") from exc


def table_exists(table_name: str, engine: Engine | None = None, **engine_kwargs) -> bool:
    engine = engine or create_db_engine(**engine_kwargs)
    return inspect(engine).has_table(table_name)
