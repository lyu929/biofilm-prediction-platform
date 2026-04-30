from __future__ import annotations

import argparse
from pathlib import Path
import warnings

try:
    from .baseline_models import train_baseline_models
    from .config import load_config
    from .data_loader import load_all_data
    from .dataset_builder import build_training_dataset
    from .eda import run_eda
except ImportError:
    from baseline_models import train_baseline_models
    from config import load_config
    from data_loader import load_all_data
    from dataset_builder import build_training_dataset
    from eda import run_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full biofilm prediction pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", choices=["database", "csv", "excel"], default="csv")
    parser.add_argument("--target", default="biofilm_thickness")
    parser.add_argument("--early_timepoints", type=int, default=10)
    parser.add_argument("--excel_path", default=None)
    parser.add_argument("--skip_mlp", action="store_true")
    parser.add_argument("--skip_eda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_config(args.config)
    data = load_all_data(source=args.source, config_path=args.config, excel_path=args.excel_path)
    dataset, X, y = build_training_dataset(
        data,
        target=args.target,
        early_timepoints=args.early_timepoints,
        config_path=args.config,
        save=True,
    )
    if not args.skip_eda:
        run_eda(data, dataset=dataset, target=args.target, config_path=args.config)

    metrics = train_baseline_models(X, y, target=args.target, config_path=args.config)
    print("Baseline metrics:")
    print(metrics.to_string(index=False))

    if not args.skip_mlp:
        try:
            from .train_mlp import train_mlp_model
        except ImportError:
            from train_mlp import train_mlp_model
        try:
            train_mlp_model(X, y, target=args.target, config_path=args.config)
            print("MLP training complete.")
        except Exception as exc:
            warnings.warn(f"MLP training skipped: {exc}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
