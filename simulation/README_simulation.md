# Simulation Workflow

This folder provides an export-oriented Python runner based on the existing
Artistoo/near-wall biofilm simulation in `src/main.js` and the analytic
Poiseuille flow server in `src/ns_simple.py`.

## Run one simulation

```bash
python simulation/run_simulation.py --config config.yaml --export csv
```

Override parameters from the command line:

```bash
python simulation/run_simulation.py \
  --flow_rate 0.5 \
  --adhesion_wall 0.8 \
  --runtime 500 \
  --export_interval 10 \
  --export csv
```

CSV files are written to `data/raw/`:

- `simulations.csv`
- `cells.csv`
- `clusters.csv`
- `flow_field.csv`
- `simulation_summary.csv`

## Run a parameter sweep

```bash
python simulation/run_parameter_sweep.py --config config.yaml --export csv
```

For a quick test:

```bash
python simulation/run_parameter_sweep.py --config config.yaml --max_runs 3 --runtime 50 --export csv
```

The example grid in `config.yaml` includes:

```yaml
flow_rate: [0.1, 0.5, 1.0]
adhesion_wall: [0.2, 0.5, 0.8]
adhesion_cell: [0.3, 0.6]
diffusion_rate: [0.01, 0.05]
qs_threshold: [0.3, 0.6]
eps_rate: [0.01, 0.03]
random_seed: [1, 2, 3]
```

Each parameter combination plus each random seed creates one unique
`simulation_id`. If 50 parameter sets are run with 5 seeds each, the sweep
creates 250 `simulation_id` values. With 100 exported timepoints and 500 cells
per timepoint, the `cells` table will contain 12,500,000 rows.

## Make wall attachment easier

Gray attached cells appear more often when wall contact and adhesion are strong
and local shear is low. Tune these fields in `config.yaml`:

```yaml
simulation:
  adhesion_wall: 0.8
  attachment_distance: 8.0
  shear_scale: 0.8
  flow_rate: 0.1
```

Attachment uses:

```text
attachment_probability = adhesion_wall * exp(-local_shear / shear_scale)
```

## Write to database

Configure `config.yaml`, then run:

```bash
python simulation/run_simulation.py --config config.yaml --export database
```

or:

```bash
python simulation/run_parameter_sweep.py --config config.yaml --export both
```

The database schema is created automatically by `src/database_init.py`.

## View data in DBeaver

1. Create a MySQL or PostgreSQL connection in DBeaver using the same values in
   `config.yaml`.
2. Run a simulation or sweep with `--export database`.
3. Refresh the schema in DBeaver.
4. Inspect the tables: `simulations`, `cells`, `clusters`, `flow_field`, and
   `simulation_summary`.
