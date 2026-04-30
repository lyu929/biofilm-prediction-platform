from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

try:
    from .config import PROJECT_ROOT, TABLE_NAMES, load_config
    from .database import create_db_engine
except ImportError:
    from config import PROJECT_ROOT, TABLE_NAMES, load_config
    from database import create_db_engine


DROP_ORDER = ["cells", "clusters", "flow_field", "simulation_summary", "simulations"]


def create_database_if_not_exists(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> None:
    cfg = load_config(config_path) if db_config is None else {"database": db_config}
    db = cfg.get("database", {})
    db_type = (db.get("type") or "mysql").lower()
    database = db.get("database")
    if not database:
        raise ValueError("Database name is missing from config.")

    if db_type in {"mysql", "mariadb"}:
        engine = create_db_engine(config_path=config_path, db_config=db, include_database=False)
        with engine.begin() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{database}`"))
        return

    if db_type in {"postgres", "postgresql"}:
        maintenance = dict(db)
        maintenance["database"] = db.get("maintenance_database", "postgres")
        engine = create_db_engine(db_config=maintenance, include_database=True)
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database},
            ).scalar()
        if not exists:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text(f'CREATE DATABASE "{database}"'))
        return

    raise ValueError(f"Unsupported database type: {db_type}")


def _split_sql(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current).rstrip(";"))
            current = []
    if current:
        statements.append("\n".join(current))
    return statements


def create_tables(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
    schema_path: str | Path | None = None,
) -> None:
    create_database_if_not_exists(config_path=config_path, db_config=db_config)
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    schema_file = Path(schema_path) if schema_path else PROJECT_ROOT / "src" / "schema.sql"
    sql = schema_file.read_text(encoding="utf-8")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in _split_sql(sql):
            try:
                conn.execute(text(statement))
            except SQLAlchemyError as exc:
                msg = str(exc).lower()
                if "already exists" in msg or "duplicate" in msg:
                    continue
                raise


def reset_tables(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> None:
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    with engine.begin() as conn:
        for table in DROP_ORDER:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    create_tables(config_path=config_path, db_config=db_config)


def test_connection(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> bool:
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


def show_table_counts(
    config_path: str | Path | None = None,
    db_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    engine = create_db_engine(config_path=config_path, db_config=db_config)
    inspector = inspect(engine)
    rows = []
    with engine.connect() as conn:
        for table in TABLE_NAMES:
            if not inspector.has_table(table):
                rows.append({"table": table, "rows": 0, "exists": False})
                continue
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            rows.append({"table": table, "rows": int(count), "exists": True})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    create_tables()
    print(show_table_counts().to_string(index=False))
