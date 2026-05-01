from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.export_simulation_data import combine_results, save_results_to_csv, write_results_to_database
from simulation.run_simulation import run_single_simulation
from simulation.simulation_config import (
    BiofilmSimulationConfig,
    expand_parameter_grid,
    parameter_grid_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameter sweep for biofilm simulations.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--export", choices=["csv", "database", "both", "none"], default="csv")
    parser.add_argument("--max_runs", type=int, default=None, help="Limit runs for quick smoke tests.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle expanded grid before applying --max_runs.")
    parser.add_argument("--shuffle_seed", type=int, default=1)
    parser.add_argument(
        "--skip_existing_csv",
        action="store_true",
        help="Skip parameter/runtime/seed combinations already present in output simulations.csv.",
    )
    parser.add_argument("--runtime", type=int, default=None)
    parser.add_argument("--export_interval", type=int, default=None)
    return parser.parse_args()


def _config_key(cfg: BiofilmSimulationConfig) -> tuple:
    return (
        round(float(cfg.flow_rate), 12),
        round(float(cfg.adhesion_wall), 12),
        round(float(cfg.adhesion_cell), 12),
        round(float(cfg.diffusion_rate), 12),
        round(float(cfg.signal_decay), 12),
        round(float(cfg.qs_threshold), 12),
        round(float(cfg.division_rate), 12),
        round(float(cfg.eps_rate), 12),
        int(cfg.runtime),
        int(cfg.random_seed),
    )


def _existing_csv_keys(output_dir: str | Path) -> set[tuple]:
    simulations_csv = Path(output_dir) / "simulations.csv"
    if not simulations_csv.exists():
        return set()

    keys = set()
    with simulations_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                keys.add(
                    (
                        round(float(row["flow_rate"]), 12),
                        round(float(row["adhesion_wall"]), 12),
                        round(float(row["adhesion_cell"]), 12),
                        round(float(row["diffusion_rate"]), 12),
                        round(float(row["signal_decay"]), 12),
                        round(float(row["qs_threshold"]), 12),
                        round(float(row["division_rate"]), 12),
                        round(float(row["eps_rate"]), 12),
                        int(float(row["runtime"])),
                        int(float(row["random_seed"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return keys


def run_parameter_sweep(
    config_path: str | Path = "config.yaml",
    output_dir: str | Path = "data/raw",
    export: str = "csv",
    max_runs: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 1,
    skip_existing_csv: bool = False,
    runtime: int | None = None,
    export_interval: int | None = None,
):
    grid = parameter_grid_from_config(config_path)
    base = BiofilmSimulationConfig()
    if runtime is not None:
        base.runtime = runtime
    if export_interval is not None:
        base.export_interval = export_interval
    configs = expand_parameter_grid(grid, base_config=base)
    if skip_existing_csv:
        existing = _existing_csv_keys(output_dir)
        before_count = len(configs)
        configs = [cfg for cfg in configs if _config_key(cfg) not in existing]
        print(f"Skipping {before_count - len(configs)} existing CSV parameter/runtime/seed combinations.")
    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(configs)
    if max_runs is not None:
        configs = configs[:max_runs]

    all_results = []
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] seed={cfg.random_seed} flow={cfg.flow_rate} adhesion_wall={cfg.adhesion_wall}")
        result = run_single_simulation(cfg)
        all_results.append(result)
        if export in {"csv", "both"}:
            save_results_to_csv(result, output_dir, append=True)
        if export in {"database", "both"}:
            write_results_to_database(result, config_path=config_path, on_existing="replace")

    return combine_results(all_results)


def main() -> None:
    args = parse_args()
    combined = run_parameter_sweep(
        config_path=args.config,
        output_dir=args.output,
        export=args.export,
        max_runs=args.max_runs,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        skip_existing_csv=args.skip_existing_csv,
        runtime=args.runtime,
        export_interval=args.export_interval,
    )
    print("Parameter sweep complete.")
    for table, df in combined.items():
        print(f"{table}: {len(df):,} rows")


if __name__ == "__main__":
    main()
