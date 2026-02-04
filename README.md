# Soil Moisture Modeling from Satellite Indices (Benchmark Suite)

A data-driven benchmarking workflow to model **surface soil moisture proxy (0–10 cm)** from **remote sensing predictors** using multiple ML/DS models and **group-aware validation**. The main script logs **timings per step/model/fold** and writes reproducible CSV outputs.

---

## What this repository does

- **Target (default):** `ka010` — TDR dielectric constant (0–10 cm)
- **Predictors:** “remote” columns from `MEAN_B02` to the last column in the combined CSV
- **Validation (two regimes):**
  - `leave_one_day_out` grouped by `Dayn` (time-robust evaluation)
  - `leave_one_site_out` grouped by `SiteTerm` (site-robust evaluation; see note below)
- **Models (7):**
  - `Ridge`
  - `ElasticNet`
  - `PLSR`
  - `SVR_RBF`
  - `GPR`
  - `RandomForest`
  - `HistGB`
- **Outputs:**
  - `fold_metrics_<outer_cv>.csv`
  - `predictions_<outer_cv>.csv`
  - `summary_<outer_cv>.csv`
  - `run_log.txt` (timing + progress)
  - optional `permutation_importance_<model>.csv`

---

## Suggested repository layout

```
.
├── soil_workflow_time.py
├── data/
│   ├── zeszyt1.csv
│   └── bazaIndeksowRS_2T_1.xlsx
└── outputs/
```

> If your files are elsewhere, either (a) edit the defaults inside the script, or (b) pass paths via command-line arguments.

---

## Data format expectations

Your combined CSV should include at least:

### Required identifiers
- `Dayn` — acquisition day/group id
- `SiteTerm` — site label

### Field measurements (targets / auxiliaries)
- `ka010` — TDR dielectric constant (0–10 cm)
- `eps010` — soil electrical conductivity (0–10 cm)
- `HeSpec` — species height (cm)
- `GWL` — ground water level from soil surface (cm)
- `NDVI` — field NDVI (GreenSeeker)

### Remote predictors block
- Remote predictors **must start at** a column named `MEAN_B02`
- All columns from `MEAN_B02` to the end are treated as remote predictors

The CSV is expected to be **semicolon-separated** (`;`). The script trims whitespace in column names.

---

## Quickstart

### 1) Install dependencies
Required packages:
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `openpyxl`

If you use ArcGIS Pro Python, prefer installing into the ArcGIS conda environment.

### 2) Run (using defaults inside the script)
If you edited defaults in the script (BASE_DIR / DEFAULT paths), you can run with no arguments:

```bash
python soil_workflow_time.py
```

### 3) Run (recommended: explicit arguments)
```bash
python soil_workflow_time.py   --data_csv "E:\pythonProject\Soil_project\zeszyt1.csv"   --outdir   "E:\pythonProject\Soil_project\outputs"   --target   ka010
```

Optional metadata filter (uses the Excel index catalogue):
```bash
python soil_workflow_time.py   --data_csv "…\zeszyt1.csv"   --index_meta_xlsx "…\bazaIndeksowRS_2T_1.xlsx"   --domain_filter "moisture"   --outdir "…\outputs"
```

---

## Outputs

All outputs are written to `--outdir`:

- `summary_leave_one_day_out.csv`  
  Model ranking & average performance when holding out entire days (`Dayn`).

- `summary_leave_one_site_out.csv`  
  Model ranking & average performance when holding out site groups (`SiteTerm`).

- `predictions_*.csv`  
  Out-of-fold predictions per record (useful for pooled metrics and plots).

- `fold_metrics_*.csv`  
  Per-fold metrics (includes `fit_seconds` and `fold_seconds`).

- `run_log.txt`  
  Human-readable log with timestamps, per-model and per-fold timings.

---

## Important note on “leave-one-site-out”

By default, grouping uses `SiteTerm`. If `SiteTerm` encodes **site + replicate** (e.g., `SITE_..._rep`), you can get tiny test sets and unstable fold metrics.

If you want a true site-level CV (e.g., ~12 sites), derive a stable site id such as:

```python
df2["SiteID"] = df2["SiteTerm"].astype(str).str.rsplit("_", n=1).str[0]
groups_site = df2["SiteID"].astype(str).to_numpy()
```

Then use `groups_site` in the site-based CV.

---

## Reproducibility tips

- Keep exact run commands in a `runs.txt`
- Commit:
  - `soil_workflow_time.py`
  - `summary_*.csv`
  - `run_log.txt`
- Record environment versions:
  - `python --version`
  - `python -c "import sklearn; print(sklearn.__version__)"`

---

## Troubleshooting

### ArcGIS Pro + PyCharm: sklearn imported from Roaming site-packages
The script includes a guard to disable user-site packages (`PYTHONNOUSERSITE=1`) to avoid importing scikit-learn from:
`C:\Users\<user>\AppData\Roaming\Python\...`

If you still see Roaming imports, set in PyCharm Run Configuration:
```
PYTHONNOUSERSITE=1
```

### Runs take a long time
This workflow performs **nested CV + grid search** for multiple models. To speed up:
- run only 2–3 models first (e.g., Ridge/ElasticNet/PLSR),
- run only one outer CV first (day-out OR site-out),
- reduce hyperparameter grids.

---
