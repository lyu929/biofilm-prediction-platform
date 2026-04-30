# Biofilm Simulation Data Analysis and Prediction Platform

This repository organizes an existing biofilm simulation workflow into a
GitHub-ready research project for simulation export, relational database
storage, feature engineering, machine learning, deep learning, and interactive
prediction.

The original simulation files are preserved in `src/main.js`,
`src/artistoo-worker.js`, `src/ns_simple.py`, and related files. The new Python
runner in `simulation/` keeps the same near-wall biofilm logic and adds stable
data export, batch parameter sweeps, database writing, and bug fixes for wall
attachment, QS/EPS behavior, division bookkeeping, and standardized table
outputs.

## Research Background

Biofilm formation depends on flow rate, cell-wall adhesion, cell-cell adhesion,
signal diffusion and decay, quorum-sensing thresholds, cell division, and EPS
production. A single simulation can generate many time-resolved cell and
cluster records. A parameter sweep with repeated random seeds naturally produces
large training datasets for predicting final biofilm morphology from early
simulation behavior.

## Database Schema

The platform uses five standard tables:

- `simulations`: one row per simulation input parameter set.
- `cells`: cell-level time series, one row per cell per exported timepoint.
- `clusters`: cluster-level time series, one row per cluster per exported timepoint.
- `flow_field`: spatial flow-grid records for each simulation.
- `simulation_summary`: final outcome labels for each simulation.

`simulation_id` links all tables. `cells` and `clusters` also include
`timepoint`.

The schema lives in `src/schema.sql` and can be created automatically:

```bash
python src/database_init.py
```

## Why Simulation Creates Large Data

One `simulation_id` represents one full run. Each run can contain many exported
timepoints, and each timepoint can contain hundreds or thousands of cells. For
example:

```text
50 parameter sets x 5 random seeds = 250 simulations
250 simulations x 100 timepoints x 500 cells = 12,500,000 cell rows
```

## Parameter Sweep

A parameter sweep runs all combinations of selected parameters and random seeds.
Each combination creates a new `simulation_id`.

Example grid in `config.yaml`:

```yaml
flow_rate: [0.1, 0.5, 1.0]
adhesion_wall: [0.2, 0.5, 0.8]
adhesion_cell: [0.3, 0.6]
diffusion_rate: [0.01, 0.05]
qs_threshold: [0.3, 0.6]
eps_rate: [0.01, 0.03]
random_seed: [1, 2, 3]
```

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
├── models/
├── outputs/
│   ├── figures/
│   ├── results/
│   └── predictions/
├── simulation/
│   ├── README_simulation.md
│   ├── run_simulation.py
│   ├── run_parameter_sweep.py
│   ├── simulation_config.py
│   ├── export_simulation_data.py
│   └── bugfix_notes.md
├── src/
│   ├── config.py
│   ├── database.py
│   ├── database_init.py
│   ├── database_writer.py
│   ├── schema.sql
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── dataset_builder.py
│   ├── eda.py
│   ├── baseline_models.py
│   ├── mlp_model.py
│   ├── train_mlp.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── visualize.py
│   └── main.py
├── app/
│   └── streamlit_app.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

Existing browser/FEM simulation files remain in `src/` as legacy simulation
assets.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Database Configuration

Edit `config.yaml`:

```yaml
database:
  type: mysql
  host: localhost
  port: 3306
  user: root
  password: your_password
  database: biofilm_db
```

PostgreSQL is also supported:

```yaml
database:
  type: postgresql
  host: localhost
  port: 5432
  user: postgres
  password: your_password
  database: biofilm_db
```

The project uses SQLAlchemy with `pymysql` for MySQL and `psycopg2-binary` for
PostgreSQL.

## DBeaver Workflow

1. Create a MySQL or PostgreSQL connection in DBeaver.
2. Use the same host, port, user, password, and database from `config.yaml`.
3. Run table initialization:

   ```bash
   python src/database_init.py
   ```

4. Run a simulation with database export.
5. Refresh the DBeaver schema and inspect the five standard tables.

## Run One Simulation

```bash
python simulation/run_simulation.py --config config.yaml --export csv
```

Write directly to the database:

```bash
python simulation/run_simulation.py --config config.yaml --export database
```

Write both CSV and database:

```bash
python simulation/run_simulation.py --config config.yaml --export both
```

Tune attachment:

```bash
python simulation/run_simulation.py \
  --adhesion_wall 0.8 \
  --flow_rate 0.1 \
  --runtime 500 \
  --export_interval 10
```

## Run Parameter Sweep

```bash
python simulation/run_parameter_sweep.py --config config.yaml --export csv
```

Quick smoke test:

```bash
python simulation/run_parameter_sweep.py --config config.yaml --max_runs 3 --runtime 50 --export csv
```

## CSV and Excel Import

Default CSV paths:

- `data/raw/simulations.csv`
- `data/raw/cells.csv`
- `data/raw/clusters.csv`
- `data/raw/flow_field.csv`
- `data/raw/simulation_summary.csv`

Excel import supports one workbook with sheets named:

- `simulations`
- `cells`
- `clusters`
- `flow_field`
- `simulation_summary`

## Full Pipeline

Run from CSV:

```bash
python src/main.py --source csv --target biofilm_thickness --early_timepoints 10
```

Run from database:

```bash
python src/main.py --source database --target roughness --early_timepoints 10
```

The pipeline:

1. Loads data.
2. Builds one-row-per-`simulation_id` features.
3. Saves `data/processed/training_dataset.csv`, `X.csv`, and `y.csv`.
4. Runs EDA plots.
5. Trains baseline models.
6. Trains the PyTorch MLP when enough samples exist.

## Models

Baseline models:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Deep learning model:

- PyTorch MLP: `input_dim -> 128 -> 64 -> 32 -> output_dim`
- ReLU activations
- Dropout(0.2)
- Adam optimizer
- MSE loss
- Early stopping

Metrics:

- MAE
- RMSE
- R2

## Prediction

Predict from parameters only:

```bash
python src/predict.py \
  --target biofilm_thickness \
  --flow_rate 0.5 \
  --adhesion_wall 0.8 \
  --adhesion_cell 0.6 \
  --diffusion_rate 0.05 \
  --signal_decay 0.01 \
  --qs_threshold 0.6 \
  --division_rate 0.01 \
  --eps_rate 0.02 \
  --runtime 300
```

When early cell or cluster data is not provided, missing early-timepoint
features are filled from the saved training feature template.

For early-data prediction, use the Streamlit app or call
`predict_from_early_data()` in `src/predict.py`.

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Pages:

- Project Intro
- Database Connection
- Simulation Data Import
- Data Analysis
- Model Training
- New Parameter Prediction
- Early Data Prediction
- Model Visualization

## Output Files

Data:

- `data/raw/*.csv`: simulation-exported raw tables.
- `data/processed/training_dataset.csv`: modeling dataset.
- `data/processed/X.csv`: feature matrix.
- `data/processed/y.csv`: selected target.

Models:

- `models/*_linear_regression.joblib`
- `models/*_ridge_regression.joblib`
- `models/*_random_forest.joblib`
- `models/*_gradient_boosting.joblib`
- `models/baseline_best_<target>.joblib`
- `models/mlp_model.pt`
- `models/mlp_scaler.joblib`
- `models/feature_template_<target>.joblib`

Results and figures:

- `outputs/results/baseline_metrics.csv`
- `outputs/results/mlp_training_history.csv`
- `outputs/figures/pred_vs_true_baseline.png`
- `outputs/figures/mlp_loss_curve.png`
- EDA and feature-importance figures

## Notes on Existing Simulation Code

The browser simulation files are not deleted or replaced. The Python runner is
an export-oriented companion that keeps the same modeling assumptions and makes
the data pipeline reliable for database storage and ML training. Details are in
`simulation/bugfix_notes.md`.
