CREATE TABLE IF NOT EXISTS simulations (
    simulation_id VARCHAR(64) PRIMARY KEY,
    flow_rate DOUBLE PRECISION,
    adhesion_wall DOUBLE PRECISION,
    adhesion_cell DOUBLE PRECISION,
    diffusion_rate DOUBLE PRECISION,
    signal_decay DOUBLE PRECISION,
    qs_threshold DOUBLE PRECISION,
    division_rate DOUBLE PRECISION,
    eps_rate DOUBLE PRECISION,
    runtime INTEGER,
    random_seed INTEGER,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cells (
    cell_id VARCHAR(64),
    simulation_id VARCHAR(64) NOT NULL,
    timepoint INTEGER NOT NULL,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    state VARCHAR(32),
    volume DOUBLE PRECISION,
    local_signal DOUBLE PRECISION,
    cluster_id VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id VARCHAR(64),
    simulation_id VARCHAR(64) NOT NULL,
    timepoint INTEGER NOT NULL,
    cluster_size INTEGER,
    center_x DOUBLE PRECISION,
    center_y DOUBLE PRECISION,
    density DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS flow_field (
    simulation_id VARCHAR(64) NOT NULL,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    velocity_x DOUBLE PRECISION,
    velocity_y DOUBLE PRECISION,
    shear DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS simulation_summary (
    simulation_id VARCHAR(64) PRIMARY KEY,
    final_cell_count INTEGER,
    biofilm_area DOUBLE PRECISION,
    biofilm_thickness DOUBLE PRECISION,
    surface_coverage DOUBLE PRECISION,
    mean_cluster_size DOUBLE PRECISION,
    max_cluster_size DOUBLE PRECISION,
    cluster_count INTEGER,
    roughness DOUBLE PRECISION,
    qs_active_ratio DOUBLE PRECISION,
    eps_fraction DOUBLE PRECISION,
    time_to_first_cluster INTEGER,
    time_to_stable_biofilm INTEGER
);

CREATE INDEX idx_simulations_simulation_id ON simulations (simulation_id);
CREATE INDEX idx_cells_simulation_id ON cells (simulation_id);
CREATE INDEX idx_cells_timepoint ON cells (timepoint);
CREATE INDEX idx_clusters_simulation_id ON clusters (simulation_id);
CREATE INDEX idx_clusters_timepoint ON clusters (timepoint);
CREATE INDEX idx_flow_field_simulation_id ON flow_field (simulation_id);
CREATE INDEX idx_simulation_summary_simulation_id ON simulation_summary (simulation_id);
