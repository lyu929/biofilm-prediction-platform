# Thesis Notes

This document summarizes how the project can support the Methods, Results, and Discussion sections of a thesis or final defense.

## Possible Thesis Title

Biofilm Simulation Data Analysis and Machine Learning Prediction Platform Based on Parameter Sweeps and Early-Timepoint Features

## Research Goal

The project studies whether final biofilm morphology can be predicted from:

- initial simulation parameters
- early cell-level dynamics
- early cluster-level dynamics
- flow and shear-field statistics

The prediction targets are:

- `biofilm_thickness`
- `surface_coverage`
- `roughness`
- `eps_fraction`
- `cluster_count`
- `time_to_stable_biofilm`

## Methods Section

### Simulation Data Generation

Biofilm simulations are run under different parameter combinations, including flow rate, wall adhesion, cell-cell adhesion, signal diffusion/decay, QS threshold, cell division rate, EPS production rate, runtime, and random seed.

Each simulation run is assigned a unique `simulation_id`. The simulation exports:

- input parameters to `simulations`
- cell-level time-series records to `cells`
- cluster-level time-series records to `clusters`
- flow-grid records to `flow_field`
- final biofilm outcomes to `simulation_summary`

Parameter sweep and repeated random seeds increase the diversity and size of the dataset.

### Feature Extraction

The model uses biofilm-level features, not raw cell rows as samples. Cell and cluster data are aggregated by `simulation_id` and `timepoint`.

Cell-derived features include:

- cell count
- total biomass
- mean/max volume
- local signal statistics
- planktonic/attached/active/inactive/QS/EPS ratios
- early biofilm thickness
- early surface coverage
- early roughness
- spatial spread in x/y

Cluster-derived features include:

- cluster count
- mean/max cluster size
- cluster density
- cluster center and spread

Flow-derived features include:

- mean/max velocity
- mean/max shear
- low/high shear fractions

### Modeling

Traditional ML models:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Deep learning model:

- PyTorch MLP
- hidden layers: 128, 64, 32
- ReLU activation
- BatchNorm
- Dropout
- Adam optimizer
- MSE loss
- early stopping

Evaluation metrics:

- MAE
- RMSE
- R2

## Results Section

Recommended figures/tables:

- table row counts and data quality summary
- target distribution plot
- correlation heatmap
- parameter vs target scatter/boxplots
- cell_count over time
- biofilm_thickness over time
- attached_ratio over time
- qs_active_ratio over time
- eps_producing_ratio over time
- cluster_count over time
- mean_cluster_size over time
- biofilm cell-state snapshot
- local signal heatmap
- flow/shear field plot
- model metric comparison table
- pred-vs-true plot
- residual plot
- feature importance and permutation importance
- ablation study plot

## Discussion Section

Possible interpretation themes:

- Higher wall adhesion should generally increase early attachment and may increase final surface coverage.
- Higher flow rate can increase shear, which may reduce stable attachment depending on wall adhesion.
- QS threshold and signal decay influence how quickly cells become QS-active.
- EPS production affects matrix formation, roughness, and stability.
- Early-timepoint features often improve prediction beyond parameters alone because they capture realized stochastic behavior.

## Limitations

Current limitations to discuss honestly:

- The Python simulation runner is export-oriented and simplified compared with fully coupled biofilm-fluid models.
- Flow field is summarized statistically and is not a full Navier-Stokes biofilm deformation solver.
- MLP uses tabular early-timepoint features, not raw spatial images or graphs.
- Predictions are reliable mainly inside the parameter ranges represented in the simulation dataset.
- If the parameter sweep is too small or too regular, model metrics may overestimate real generalization.

## Future Work

Possible extensions:

- LSTM/GRU/Transformer models for explicit time-series prediction.
- CNN models using spatial biofilm snapshots as images.
- Graph Neural Networks for cell-cell interaction graphs.
- SHAP or other local explainability methods.
- More realistic Navier-Stokes coupling and biofilm-fluid feedback.
- Experimental validation with real biofilm microscopy or reactor data.

## Defense Talking Points

Short explanation:

This platform converts high-volume biofilm simulation output into a reproducible data science workflow. The database stores parameters, process dynamics, flow fields, and final outcomes. Feature engineering aggregates cell-level and cluster-level observations into one simulation-level sample, allowing ML and DL models to predict final biofilm properties from early simulation behavior or new parameter combinations.

Why it matters:

- It reduces the need to run long simulations for every new parameter set.
- It reveals which parameters and early dynamics most influence biofilm formation.
- It provides a repeatable pipeline from simulation to prediction and visualization.

What to show in the demo:

1. DBeaver tables with many automatically generated rows.
2. Streamlit data analysis page with EDA plots.
3. Model training metrics.
4. Prediction from a new parameter combination.
5. Feature importance or ablation plot for interpretation.
