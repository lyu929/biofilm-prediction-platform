from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings
import ast

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

TABLE_NAMES = [
    "simulations",
    "cells",
    "clusters",
    "flow_field",
    "simulation_summary",
]

TARGET_COLUMNS = [
    "biofilm_thickness",
    "surface_coverage",
    "roughness",
    "eps_fraction",
    "cluster_count",
    "time_to_stable_biofilm",
]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return [v.strip() for v in value.strip("[]").split(",") if v.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def simple_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for this project's simple config.yaml format."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = raw.split(" #", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def load_yaml_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return simple_yaml_load(text)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and create standard output directories."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        warnings.warn(f"Config file not found at {path}. Using in-code defaults.")
        cfg: dict[str, Any] = {}
    else:
        cfg = load_yaml_file(path)

    cfg.setdefault("paths", {})
    cfg.setdefault("database", {})
    cfg.setdefault("simulation", {})
    cfg.setdefault("modeling", {})
    ensure_directories(cfg)
    return cfg


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(str, parts))


def get_path(name: str, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    defaults = {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "models": "models",
        "figures": "outputs/figures",
        "results": "outputs/results",
        "predictions": "outputs/predictions",
    }
    rel = cfg.get("paths", {}).get(name, defaults.get(name, name))
    path = project_path(rel)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directories(cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or {}
    for key in ["raw_data", "processed_data", "models", "figures", "results", "predictions"]:
        rel = cfg.get("paths", {}).get(key)
        if rel is None:
            rel = {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "models": "models",
                "figures": "outputs/figures",
                "results": "outputs/results",
                "predictions": "outputs/predictions",
            }[key]
        project_path(rel).mkdir(parents=True, exist_ok=True)


def warn_missing_columns(df, required: list[str], table_name: str) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(f"{table_name} is missing columns: {missing}. Missing values will be filled.")
    return missing


def numeric_fill_value(series):
    if series is None or len(series) == 0:
        return 0.0
    median = series.median()
    if median != median:
        return 0.0
    return float(median)
