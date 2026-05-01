# Browser Simulation to DBeaver Database Workflow

This workflow uses the existing browser/Artistoo simulation in `src/main.js`.
It does not require CSV export.

## What Runs

1. `src/index.html` opens the original browser simulation.
2. `src/main.js` advances the biofilm model and sends snapshots every
   `DB_EXPORT_CONFIG.exportInterval` MCS.
3. `src/ns_simple.py` receives those snapshots through WebSocket, computes the
   flow response, and writes records to the database with SQLAlchemy.
4. DBeaver shows the database tables updating as the simulation runs.

## Configure The Database

Edit the `database` section in `config.yaml` so it matches your DBeaver
connection:

```yaml
database:
  type: mysql
  host: localhost
  port: 3306
  user: root
  password: your_password
  database: biofilm_db
```

You can also avoid saving the password in `config.yaml`:

```bash
export BIOFILM_DB_PASSWORD="your_password"
```

Other supported overrides are:

- `BIOFILM_DB_TYPE`
- `BIOFILM_DB_HOST`
- `BIOFILM_DB_PORT`
- `BIOFILM_DB_USER`
- `BIOFILM_DB_NAME`

## Create Tables

From the project root:

```bash
.venv/bin/python src/database_init.py
```

Refresh DBeaver. You should see:

- `simulations`
- `cells`
- `clusters`
- `flow_field`
- `simulation_summary`

## Start The Database/Flow Server

Terminal 1, from the project root:

```bash
.venv/bin/python src/ns_simple.py
```

This starts:

- flow-force WebSocket server: `ws://localhost:8765`
- database writer bridge: MySQL/PostgreSQL through SQLAlchemy

## Start The Browser Simulation

Terminal 2, from the project root:

```bash
cd src
../.venv/bin/python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

Every browser reload creates a new `simulation_id`. Every 10 MCS by default,
the simulation appends new `cells` and `clusters` rows and updates
`simulation_summary`. The `flow_field` table is written once per simulation.

## Where To Change Export Settings

Open `src/main.js` and edit:

```javascript
const DB_EXPORT_CONFIG = {
    enabled: true,
    exportInterval: 10,
    params: {
        flow_rate: 0.5,
        adhesion_wall: 0.6,
        adhesion_cell: 0.5
    }
};
```

For less data, increase `exportInterval`, for example `50`.
For more detailed time series, reduce it to `5` or `10`.

## DBeaver Checks

After the browser simulation starts, refresh the `biofilm_db` tables and run:

```sql
SELECT COUNT(*) FROM simulations;
SELECT COUNT(*) FROM cells;
SELECT COUNT(*) FROM clusters;
SELECT COUNT(*) FROM flow_field;
SELECT COUNT(*) FROM simulation_summary;
```

Inspect one simulation:

```sql
SELECT * FROM simulations ORDER BY created_at DESC LIMIT 5;

SELECT timepoint, COUNT(*) AS cell_rows
FROM cells
GROUP BY timepoint
ORDER BY timepoint;

SELECT *
FROM simulation_summary
ORDER BY simulation_id DESC
LIMIT 10;
```

## Use The Database In The Platform

After data is in DBeaver, build training data directly from the database:

```bash
.venv/bin/python src/main.py --source database --target biofilm_thickness --early_timepoints 10
```

Or start the app:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Then choose `database` as the data source in the analysis/training pages.
