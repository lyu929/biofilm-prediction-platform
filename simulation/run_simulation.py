from __future__ import annotations

import argparse
import hashlib
import math
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.export_simulation_data import save_results_to_csv, write_results_to_database
from simulation.simulation_config import BiofilmSimulationConfig, load_simulation_config


BIOFILM_STATES = {"attached", "qs_active", "eps_producing", "inactive"}


class BiofilmSimulationModel:
    """Export-oriented Python runner derived from the existing Artistoo setup.

    It preserves the documented near-wall geometry, Poiseuille-like flow,
    signal diffusion/decay, QS activation, EPS production, cell growth/division,
    and open-boundary deletion from the existing JS/Python simulation. The key
    addition is a deterministic wall-attachment update and standardized table
    export for downstream analysis.
    """

    def __init__(self, config: BiofilmSimulationConfig, simulation_id: str | None = None):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)
        self.simulation_id = simulation_id or self._make_simulation_id(config)
        self.created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.next_cell_id = 1
        self.cells: list[dict[str, Any]] = []
        self.signal = np.zeros((config.signal_grid_ny, config.signal_grid_nx), dtype=float)
        self.first_cluster_time: int | None = None
        self.stable_biofilm_time: int | None = None
        self._biofilm_count_history: list[tuple[int, int]] = []
        self._latest_cluster_ids: dict[str, str] = {}
        self.flow_field = self._build_flow_field()
        self._seed_initial_cells()

    @staticmethod
    def _make_simulation_id(config: BiofilmSimulationConfig) -> str:
        payload = repr(sorted(asdict(config).items())).encode("utf-8")
        digest = hashlib.sha1(payload + uuid.uuid4().bytes).hexdigest()[:10]
        return f"sim_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{digest}"

    def _seed_initial_cells(self) -> None:
        h = self.cfg.field_height
        for i in range(self.cfg.initial_cell_count):
            x = self.rng.uniform(2.0, min(40.0, self.cfg.field_width * 0.12))
            if i < max(2, self.cfg.initial_cell_count // 4):
                y = h - 1.0 - self.rng.uniform(1.0, self.cfg.attachment_distance * 1.5)
            else:
                y = self.rng.uniform(5.0, h - 12.0)
            self._add_cell(x, y, state="planktonic")

    def _add_cell(self, x: float, y: float, state: str, volume: float = 12.0) -> str:
        cid = str(self.next_cell_id)
        self.next_cell_id += 1
        self.cells.append(
            {
                "cell_id": cid,
                "x": float(np.clip(x, 0.0, self.cfg.field_width - 1.0)),
                "y": float(np.clip(y, 0.0, self.cfg.field_height - 2.0)),
                "state": state,
                "volume": float(volume),
                "local_signal": 0.0,
                "age": 0,
                "eps_mass": 0.0,
            }
        )
        return cid

    def _build_flow_field(self) -> pd.DataFrame:
        rows = []
        nx, ny = self.cfg.flow_grid_nx, self.cfg.flow_grid_ny
        w, h = self.cfg.field_width, self.cfg.field_height
        for j in range(ny):
            y = (j + 0.5) * h / ny
            y_from_wall = max(0.0, h - y)
            velocity_x = self.cfg.flow_rate * (y_from_wall / h)
            shear = self.cfg.flow_rate / max(h, 1.0)
            for i in range(nx):
                x = (i + 0.5) * w / nx
                rows.append(
                    {
                        "simulation_id": self.simulation_id,
                        "x": x,
                        "y": y,
                        "velocity_x": velocity_x,
                        "velocity_y": 0.0,
                        "shear": shear,
                    }
                )
        return pd.DataFrame(rows)

    def _velocity_and_shear(self, x: float, y: float) -> tuple[float, float]:
        y_from_wall = max(0.0, self.cfg.field_height - y)
        velocity_x = self.cfg.flow_rate * (y_from_wall / self.cfg.field_height)
        shear = self.cfg.flow_rate / max(self.cfg.field_height, 1.0)
        return velocity_x, shear

    def _signal_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(x / self.cfg.field_width * self.cfg.signal_grid_nx, 0, self.cfg.signal_grid_nx - 1))
        iy = int(np.clip(y / self.cfg.field_height * self.cfg.signal_grid_ny, 0, self.cfg.signal_grid_ny - 1))
        return iy, ix

    def _diffuse_signal(self) -> None:
        s = self.signal
        diffusion = float(np.clip(self.cfg.diffusion_rate, 0.0, 0.24))
        lap = (
            np.roll(s, 1, axis=0)
            + np.roll(s, -1, axis=0)
            + np.roll(s, 1, axis=1)
            + np.roll(s, -1, axis=1)
            - 4.0 * s
        )
        self.signal = np.maximum(0.0, s + diffusion * lap)
        self.signal *= max(0.0, 1.0 - self.cfg.signal_decay)
        self.signal[0, :] = 0.0

    def _produce_signal(self) -> None:
        for cell in self.cells:
            iy, ix = self._signal_index(cell["x"], cell["y"])
            rate = 0.03
            if cell["state"] in {"attached", "qs_active", "eps_producing"}:
                rate *= 1.6
            if cell["state"] in {"qs_active", "eps_producing"}:
                rate *= 2.0
            self.signal[iy, ix] += rate

    def _update_cell_signals_and_states(self) -> None:
        for cell in self.cells:
            iy, ix = self._signal_index(cell["x"], cell["y"])
            local_signal = float(self.signal[iy, ix])
            cell["local_signal"] = local_signal
            wall_distance = self.cfg.field_height - 1.0 - cell["y"]
            near_or_attached = cell["state"] in BIOFILM_STATES or wall_distance <= self.cfg.attachment_distance * 1.5
            if local_signal >= self.cfg.qs_threshold:
                if not near_or_attached:
                    cell["state"] = "active"
                elif self.rng.random() < max(0.05, self.cfg.eps_rate):
                    cell["state"] = "eps_producing"
                else:
                    cell["state"] = "qs_active"
            elif cell["state"] == "active" and local_signal < self.cfg.qs_threshold * 0.35:
                cell["state"] = "planktonic"
            elif cell["state"] in {"qs_active", "eps_producing"} and local_signal < self.cfg.qs_threshold * 0.35:
                cell["state"] = "attached" if wall_distance <= self.cfg.attachment_distance * 1.5 else "planktonic"

    def _attachment_probability(self, cell: dict[str, Any]) -> float:
        _, shear = self._velocity_and_shear(cell["x"], cell["y"])
        base = self.cfg.adhesion_wall * math.exp(-shear / max(self.cfg.shear_scale, 1e-9))
        if cell["state"] in {"qs_active", "eps_producing"}:
            base *= 1.15
        return float(np.clip(base, 0.0, 0.98))

    def _update_attachment(self) -> None:
        h = self.cfg.field_height
        for cell in self.cells:
            wall_distance = h - 1.0 - cell["y"]
            if cell["state"] in BIOFILM_STATES:
                _, shear = self._velocity_and_shear(cell["x"], cell["y"])
                detach_prob = max(0.0, shear - self.cfg.adhesion_wall * 0.02) * 0.002
                if self.rng.random() < detach_prob:
                    cell["state"] = "detached"
                continue
            if wall_distance <= self.cfg.attachment_distance:
                if self.rng.random() < self._attachment_probability(cell):
                    cell["state"] = "attached"
                    cell["y"] = min(cell["y"], h - 2.0)

    def _move_cells(self) -> None:
        for cell in self.cells:
            vx, _ = self._velocity_and_shear(cell["x"], cell["y"])
            noise_x = self.rng.normal(0.0, 0.25)
            noise_y = self.rng.normal(0.0, 0.25)
            if cell["state"] in BIOFILM_STATES:
                cell["x"] += noise_x * 0.08
                cell["y"] += noise_y * 0.04
            else:
                cell["x"] += vx + noise_x
                wall_bias = 0.03 * self.cfg.adhesion_wall
                cell["y"] += noise_y + wall_bias
            cell["x"] = float(np.clip(cell["x"], 0.0, self.cfg.field_width - 1.0))
            cell["y"] = float(np.clip(cell["y"], 0.0, self.cfg.field_height - 2.0))

    def _grow_divide_eps(self) -> None:
        new_cells: list[tuple[float, float, str, float]] = []
        for cell in self.cells:
            cell["age"] += 1
            grow = 0.015
            if cell["state"] in {"attached", "qs_active", "eps_producing"}:
                grow += 0.01
            cell["volume"] = min(30.0, cell["volume"] + grow)
            if cell["state"] == "eps_producing":
                cell["eps_mass"] += self.cfg.eps_rate

            can_divide = cell["volume"] >= 18.0 and cell["age"] >= 20
            if can_divide and len(self.cells) + len(new_cells) < self.cfg.max_cells:
                if self.rng.random() < self.cfg.division_rate:
                    child_state = cell["state"]
                    child_volume = max(6.0, cell["volume"] / 2.0)
                    cell["volume"] = child_volume
                    cell["age"] = 0
                    dx, dy = self.rng.normal(0.0, 2.0, size=2)
                    new_cells.append((cell["x"] + dx, cell["y"] + dy, child_state, child_volume))
        for x, y, state, volume in new_cells:
            self._add_cell(x, y, state=state, volume=volume)

    def _spawn_cells(self) -> None:
        prob = self.cfg.spawn_probability * max(0.1, self.cfg.flow_rate)
        if self.rng.random() >= prob or len(self.cells) >= self.cfg.max_cells:
            return
        y = self.rng.uniform(5.0, self.cfg.field_height - 12.0)
        self._add_cell(5.0, y, state="planktonic")

    def _delete_open_boundary_cells(self) -> None:
        w, h = self.cfg.field_width, self.cfg.field_height
        kept = []
        for cell in self.cells:
            at_outlet = cell["x"] >= w - 3.0
            at_open_top = cell["y"] <= 2.0
            if at_outlet or at_open_top:
                continue
            kept.append(cell)
        self.cells = kept

    def _cluster_assignments(self) -> tuple[dict[str, str], list[dict[str, Any]]]:
        biofilm_cells = [c for c in self.cells if c["state"] in BIOFILM_STATES]
        n = len(biofilm_cells)
        if n == 0:
            return {}, []
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        coords = np.array([[c["x"], c["y"]] for c in biofilm_cells])
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= self.cfg.cluster_distance * (1.0 + self.cfg.adhesion_cell):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        assignments: dict[str, str] = {}
        clusters = []
        for k, indices in enumerate(groups.values(), start=1):
            cluster_id = f"{self.simulation_id}_c{k}"
            pts = coords[indices]
            for idx in indices:
                assignments[str(biofilm_cells[idx]["cell_id"])] = cluster_id
            width = max(1.0, float(pts[:, 0].max() - pts[:, 0].min() + 1.0))
            height = max(1.0, float(pts[:, 1].max() - pts[:, 1].min() + 1.0))
            density = len(indices) / (width * height)
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": int(len(indices)),
                    "center_x": float(pts[:, 0].mean()),
                    "center_y": float(pts[:, 1].mean()),
                    "density": float(density),
                }
            )
        return assignments, clusters

    def _record_stability(self, timepoint: int, cluster_count: int) -> None:
        biofilm_count = sum(1 for c in self.cells if c["state"] in BIOFILM_STATES)
        self._biofilm_count_history.append((timepoint, biofilm_count))
        if cluster_count > 0 and self.first_cluster_time is None:
            self.first_cluster_time = timepoint
        window = self._biofilm_count_history[-5:]
        if self.stable_biofilm_time is None and len(window) == 5:
            counts = np.array([v for _, v in window], dtype=float)
            if counts.mean() > 0 and counts.std() / counts.mean() < 0.05:
                self.stable_biofilm_time = window[0][0]

    def _snapshot_cells(self, timepoint: int, assignments: dict[str, str]) -> pd.DataFrame:
        rows = []
        for cell in self.cells:
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "simulation_id": self.simulation_id,
                    "timepoint": int(timepoint),
                    "x": float(cell["x"]),
                    "y": float(cell["y"]),
                    "state": cell["state"],
                    "volume": float(cell["volume"]),
                    "local_signal": float(cell.get("local_signal", 0.0)),
                    "cluster_id": assignments.get(str(cell["cell_id"]), None),
                }
            )
        return pd.DataFrame(rows)

    def _snapshot_clusters(self, timepoint: int, clusters: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for cluster in clusters:
            row = dict(cluster)
            row["simulation_id"] = self.simulation_id
            row["timepoint"] = int(timepoint)
            rows.append(row)
        return pd.DataFrame(rows)

    def _simulation_params_frame(self) -> pd.DataFrame:
        params = {
            "simulation_id": self.simulation_id,
            "flow_rate": self.cfg.flow_rate,
            "adhesion_wall": self.cfg.adhesion_wall,
            "adhesion_cell": self.cfg.adhesion_cell,
            "diffusion_rate": self.cfg.diffusion_rate,
            "signal_decay": self.cfg.signal_decay,
            "qs_threshold": self.cfg.qs_threshold,
            "division_rate": self.cfg.division_rate,
            "eps_rate": self.cfg.eps_rate,
            "runtime": self.cfg.runtime,
            "random_seed": self.cfg.random_seed,
            "created_at": self.created_at,
        }
        return pd.DataFrame([params])

    def _summary_frame(self, final_clusters: list[dict[str, Any]]) -> pd.DataFrame:
        biofilm_cells = [c for c in self.cells if c["state"] in BIOFILM_STATES]
        final_count = len(self.cells)
        biofilm_area = float(sum(c["volume"] for c in biofilm_cells))
        if biofilm_cells:
            heights = np.array([self.cfg.field_height - c["y"] for c in biofilm_cells], dtype=float)
            thickness = float(heights.max())
            roughness = float(heights.std())
            coverage = float(len(np.unique(np.floor([c["x"] for c in biofilm_cells]).astype(int)))) / self.cfg.field_width
        else:
            thickness = 0.0
            roughness = 0.0
            coverage = 0.0
        cluster_sizes = [c["cluster_size"] for c in final_clusters]
        qs_count = sum(1 for c in self.cells if c["state"] in {"qs_active", "eps_producing"})
        eps_count = sum(1 for c in self.cells if c["state"] == "eps_producing")
        row = {
            "simulation_id": self.simulation_id,
            "final_cell_count": int(final_count),
            "biofilm_area": biofilm_area,
            "biofilm_thickness": thickness,
            "surface_coverage": coverage,
            "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "max_cluster_size": float(np.max(cluster_sizes)) if cluster_sizes else 0.0,
            "cluster_count": int(len(final_clusters)),
            "roughness": roughness,
            "qs_active_ratio": float(qs_count / final_count) if final_count else 0.0,
            "eps_fraction": float(eps_count / final_count) if final_count else 0.0,
            "time_to_first_cluster": int(self.first_cluster_time) if self.first_cluster_time is not None else -1,
            "time_to_stable_biofilm": int(self.stable_biofilm_time) if self.stable_biofilm_time is not None else -1,
        }
        return pd.DataFrame([row])

    def step(self, timepoint: int) -> tuple[dict[str, str], list[dict[str, Any]]]:
        self._produce_signal()
        self._diffuse_signal()
        self._update_cell_signals_and_states()
        self._update_attachment()
        self._move_cells()
        self._grow_divide_eps()
        self._spawn_cells()
        self._delete_open_boundary_cells()
        assignments, clusters = self._cluster_assignments()
        self._latest_cluster_ids = assignments
        self._record_stability(timepoint, len(clusters))
        return assignments, clusters

    def run(self) -> dict[str, pd.DataFrame]:
        cell_frames = []
        cluster_frames = []
        final_clusters: list[dict[str, Any]] = []
        for t in range(self.cfg.runtime + 1):
            assignments, clusters = self._cluster_assignments() if t == 0 else self.step(t)
            final_clusters = clusters
            if t % max(1, self.cfg.export_interval) == 0 or t == self.cfg.runtime:
                cell_frames.append(self._snapshot_cells(t, assignments))
                cluster_frames.append(self._snapshot_clusters(t, clusters))

        return {
            "simulations": self._simulation_params_frame(),
            "cells": pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame(),
            "clusters": pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame(),
            "flow_field": self.flow_field.copy(),
            "simulation_summary": self._summary_frame(final_clusters),
        }


def run_single_simulation(
    config: BiofilmSimulationConfig,
    simulation_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    model = BiofilmSimulationModel(config, simulation_id=simulation_id)
    return model.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one biofilm simulation and export standard tables.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--export", choices=["csv", "database", "both", "none"], default="csv")
    parser.add_argument("--simulation_id", default=None)
    parser.add_argument("--flow_rate", type=float, default=None)
    parser.add_argument("--adhesion_wall", type=float, default=None)
    parser.add_argument("--adhesion_cell", type=float, default=None)
    parser.add_argument("--diffusion_rate", type=float, default=None)
    parser.add_argument("--signal_decay", type=float, default=None)
    parser.add_argument("--qs_threshold", type=float, default=None)
    parser.add_argument("--division_rate", type=float, default=None)
    parser.add_argument("--eps_rate", type=float, default=None)
    parser.add_argument("--runtime", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--export_interval", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simulation_config(args.config)
    for key in [
        "flow_rate",
        "adhesion_wall",
        "adhesion_cell",
        "diffusion_rate",
        "signal_decay",
        "qs_threshold",
        "division_rate",
        "eps_rate",
        "runtime",
        "random_seed",
        "export_interval",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    results = run_single_simulation(cfg, simulation_id=args.simulation_id)
    if args.export in {"csv", "both"}:
        save_results_to_csv(results, args.output, append=True)
    if args.export in {"database", "both"}:
        write_results_to_database(results, config_path=args.config, on_existing="replace")
    sid = results["simulations"]["simulation_id"].iloc[0]
    print(f"Simulation complete: {sid}")
    print(results["simulation_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
