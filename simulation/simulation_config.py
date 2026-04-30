from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        from src.config import load_yaml_file
    except ImportError:
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.config import load_yaml_file

    return load_yaml_file(path)


@dataclass
class BiofilmSimulationConfig:
    flow_rate: float = 0.5
    adhesion_wall: float = 0.6
    adhesion_cell: float = 0.5
    diffusion_rate: float = 0.05
    signal_decay: float = 0.01
    qs_threshold: float = 0.6
    division_rate: float = 0.01
    eps_rate: float = 0.02
    runtime: int = 300
    random_seed: int = 1
    export_interval: int = 10
    initial_cell_count: int = 25
    field_width: float = 500.0
    field_height: float = 200.0
    attachment_distance: float = 6.0
    shear_scale: float = 0.5
    cluster_distance: float = 8.0
    spawn_probability: float = 0.05
    max_cells: int = 2500
    signal_grid_nx: int = 100
    signal_grid_ny: int = 40
    flow_grid_nx: int = 80
    flow_grid_ny: int = 40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PARAMETER_GRID = {
    "flow_rate": [0.1, 0.5, 1.0],
    "adhesion_wall": [0.2, 0.5, 0.8],
    "adhesion_cell": [0.3, 0.6],
    "diffusion_rate": [0.01, 0.05],
    "qs_threshold": [0.3, 0.6],
    "eps_rate": [0.01, 0.03],
    "random_seed": [1, 2, 3],
}


def load_simulation_config(config_path: str | Path | None = None) -> BiofilmSimulationConfig:
    if not config_path:
        return BiofilmSimulationConfig()
    path = Path(config_path)
    data = _load_yaml(path)
    sim_data = data.get("simulation", data)
    valid = BiofilmSimulationConfig.__dataclass_fields__.keys()
    kwargs = {k: v for k, v in sim_data.items() if k in valid}
    return BiofilmSimulationConfig(**kwargs)


def parameter_grid_from_config(config_path: str | Path | None = None) -> dict[str, list[Any]]:
    if not config_path:
        return DEFAULT_PARAMETER_GRID
    path = Path(config_path)
    data = _load_yaml(path)
    grid = data.get("parameter_sweep", DEFAULT_PARAMETER_GRID)
    return {k: v for k, v in grid.items() if isinstance(v, list)}


def expand_parameter_grid(
    grid: dict[str, list[Any]] | None = None,
    base_config: BiofilmSimulationConfig | None = None,
) -> list[BiofilmSimulationConfig]:
    grid = grid or DEFAULT_PARAMETER_GRID
    base = (base_config or BiofilmSimulationConfig()).to_dict()
    keys = list(grid.keys())
    configs: list[BiofilmSimulationConfig] = []
    for values in product(*[grid[k] for k in keys]):
        params = dict(base)
        params.update(dict(zip(keys, values)))
        configs.append(BiofilmSimulationConfig(**params))
    return configs
