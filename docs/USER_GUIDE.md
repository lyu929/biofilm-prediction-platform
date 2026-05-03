# User Guide

This guide explains how to use the Biofilm Simulation Data Analysis and Prediction Platform from a clean terminal, DBeaver, and the Streamlit app.

## 1. What The Platform Does

The platform turns biofilm simulation output into a machine-learning workflow:

1. Run single simulations or parameter sweeps.
2. Store simulation output in MySQL/PostgreSQL or CSV files.
3. Aggregate cell-level and cluster-level time-series data into biofilm-level features.
4. Build one modeling sample per `simulation_id`.
5. Run EDA, model training, evaluation, and prediction.
6. Use Streamlit for interactive analysis and prediction.

The key rule is: individual cells are not training samples. Cells are first aggregated by `simulation_id` and `timepoint`; then early timepoints are expanded into one row per simulation.

## 2. Installation

Run from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On this machine, you can also use:

```bash
.venv/bin/python src/main.py --help
```

## 3. Database Configuration

Edit `config.yaml`:

```yaml
database:
  type: mysql
  host: localhost
  port: 3306
  user: root
  password:
  database: biofilm_db
```

Field meanings:

- `type`: `mysql` or `postgresql`.
- `host`: usually `localhost` when the database runs on your computer.
- `port`: MySQL usually uses `3306`; PostgreSQL usually uses `5432`.
- `user`: the database username shown in DBeaver.
- `password`: the database password. Leave blank only if your local database allows it.
- `database`: the schema/database name, for example `biofilm_db`.

You can also set credentials with environment variables:

```bash
export BIOFILM_DB_USER=root
export BIOFILM_DB_PASSWORD="your_password"
export BIOFILM_DB_NAME=biofilm_db
```

## 4. DBeaver Workflow

1. Open DBeaver.
2. Create or open the MySQL/PostgreSQL connection.
3. Use the same `host`, `port`, `user`, `password`, and `database` as `config.yaml`.
4. Initialize tables:

```bash
.venv/bin/python src/database_init.py
```

5. Run a simulation or parameter sweep with database export.
6. In DBeaver, right-click `biofilm_db` and select Refresh.
7. Inspect:

- `simulations`
- `cells`
- `clusters`
- `flow_field`
- `simulation_summary`

## 5. Run One Simulation

Write output directly to the database:

```bash
.venv/bin/python simulation/run_simulation.py --config config.yaml --export database
```

Write both database and CSV:

```bash
.venv/bin/python simulation/run_simulation.py --config config.yaml --export both
```

Increase wall attachment:

```bash
.venv/bin/python simulation/run_simulation.py \
  --config config.yaml \
  --adhesion_wall 0.8 \
  --flow_rate 0.1 \
  --export_interval 10 \
  --export database
```

## 6. Run Parameter Sweep

For training data, use parameter sweep:

```bash
.venv/bin/python simulation/run_parameter_sweep.py \
  --config config.yaml \
  --shuffle \
  --shuffle_seed 20260502 \
  --max_runs 1000 \
  --runtime 100 \
  --export_interval 10 \
  --export database
```

Each parameter combination plus one random seed creates a new `simulation_id`.

Example scaling:

```text
10 parameter sets x 3 seeds x 100 timepoints x 300 cells
= 900,000 cell-level records
```

## 7. Run Full Modeling Pipeline

From database:

```bash
.venv/bin/python src/main.py \
  --source database \
  --target biofilm_thickness \
  --early_timepoints 10
```

From CSV:

```bash
.venv/bin/python src/main.py \
  --source csv \
  --target roughness \
  --early_timepoints 10
```

Optional cross-validation and tuning:

```bash
.venv/bin/python src/main.py \
  --source database \
  --target biofilm_thickness \
  --early_timepoints 10 \
  --run_cv \
  --tune
```

## 8. Start Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Main pages:

- Project Intro: explains the platform and targets.
- Database Connection: test connection, create/reset tables, preview data.
- Simulation Data Import: upload CSV/Excel and write to database.
- Data Analysis: run EDA and view plots.
- Model Training: build dataset and train baseline/MLP models.
- Cross-Validation & Tuning: K-fold CV and GridSearchCV.
- Ablation Study: compare feature groups.
- Prediction Uncertainty: bootstrap confidence intervals.
- New Parameter Prediction: predict from only simulation parameters.
- Early Data Prediction: upload early cells/clusters and predict final outcomes.
- Model Visualization: browse saved figures.

## 9. Prediction From Parameters

```bash
.venv/bin/python src/predict.py \
  --target all \
  --model_type baseline \
  --flow_rate 0.5 \
  --adhesion_wall 0.7 \
  --adhesion_cell 0.6 \
  --diffusion_rate 0.03 \
  --signal_decay 0.02 \
  --qs_threshold 0.4 \
  --division_rate 0.02 \
  --eps_rate 0.02 \
  --runtime 100 \
  --random_seed 42
```

When early cell/cluster data is not provided, the saved feature template fills missing early-timepoint columns using training-set medians or zeros.

## 10. Outputs

Data:

- `data/processed/training_dataset.csv`
- `data/processed/X.csv`
- `data/processed/y.csv`

Models:

- `models/baseline_best_<target>.joblib`
- `models/mlp_model_<target>.pt`
- `models/mlp_scaler_<target>.joblib`
- `models/feature_template_<target>.joblib`

Figures:

- `outputs/figures/<target>_distribution.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/*_over_time.png`
- `outputs/figures/biofilm_cell_state_snapshot.png`
- `outputs/figures/local_signal_heatmap.png`
- `outputs/figures/flow_shear_field.png`
- `outputs/figures/baseline_<target>_pred_vs_true.png`
- `outputs/figures/baseline_<target>_residuals.png`
- `outputs/figures/mlp_<target>_pred_vs_true.png`
- `outputs/figures/*_feature_importance.png`
- `outputs/figures/*_permutation_importance.png`
