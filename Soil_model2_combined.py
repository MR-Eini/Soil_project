#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Soil moisture modeling from satellite indices (Enhanced Benchmark)

Features:
  - Multi-target support (ka010, GWL, NDVI, eps010, HeSpec)
  - PowerTransformer (Yeo-Johnson) for target normalization
  - Recursive Feature Selection (SelectKBest) inside pipelines
  - Stacking Regressor (Hybrid Ensemble)
  - Detailed Execution Timing Log
  - OPTIONAL: Random CV (Mixing sites) via --mix_sites

Usage:
  # Robust Spatial Validation (Default):
  python soil_workflow_time.py --target ka010

  # Random Validation (Mix all sites together):
  python soil_workflow_time.py --target ka010 --mix_sites
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# IMPORTANT (ArcGIS Pro + PyCharm fix):
# Prevent importing user-site packages (Roaming\Python313) which can override the
# ArcGIS conda environment packages.
# This block MUST run before importing scikit-learn.
# -----------------------------------------------------------------------------
os.environ.setdefault("PYTHONNOUSERSITE", "1")
try:
    import site

    site.ENABLE_USER_SITE = False

    # Remove user site packages from sys.path if present
    user_site = site.getusersitepackages()
    user_sites = [user_site] if isinstance(user_site, str) else list(user_site)


    def _is_user_site(p: str) -> bool:
        ap = os.path.abspath(p).lower()
        # Common pattern on Windows
        if "\\appdata\\roaming\\python\\" in ap:
            return True
        for us in user_sites:
            if not us:
                continue
            if ap.startswith(os.path.abspath(us).lower()):
                return True
        return False


    sys.path = [p for p in sys.path if not _is_user_site(p)]
except Exception:
    pass
# -----------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge, ElasticNet, RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.inspection import permutation_importance

# --------------------------- DEFAULT PATHS -----------------------------------
# Update these if your paths differ
BASE_DIR = Path(r"E:\pythonProject\Soil_project")
DEFAULT_CSV = BASE_DIR / "zeszyt1.csv"
DEFAULT_XLSX = BASE_DIR / "bazaIndeksowRS_2T_1.xlsx"
DEFAULT_OUTDIR = BASE_DIR / "outputs"


# --------------------------- LOGGING & TIMING --------------------------------

def log(msg: str, log_file: Path | None = None):
    """Prints message with timestamp and optionally appends to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")


def fmt_seconds(s: float) -> str:
    """Formats seconds into mm:ss.ss string."""
    m, s_rem = divmod(s, 60)
    return f"{int(m)}m {s_rem:.2f}s"


# --------------------------- DATA PREP ---------------------------------------

def read_dataset(csv_path: Path) -> pd.DataFrame:
    # your file is semicolon-separated
    df = pd.read_csv(csv_path, sep=";")
    # strip accidental spaces in column names (e.g., "SEI ")
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_columns(df: pd.DataFrame):
    # mandatory identifiers
    id_cols = ["Dayn", "SiteTerm"]
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # field measurements (targets or auxiliary)
    # Check for at least one of these to ensure file structure is correct
    possible_targets = ["ka010", "eps010", "HeSpec", "GWL", "NDVI"]
    found_targets = [c for c in possible_targets if c in df.columns]
    if not found_targets:
        raise ValueError(f"No standard target columns found. Looked for: {possible_targets}")

    # ignore block H..M in your description (likely rb..teta2)
    ignore_candidates = ["rb", "pd", "Po", "alfa", "teta1", "teta2"]
    ignore_cols = [c for c in ignore_candidates if c in df.columns]

    # remote block starts at MEAN_B02
    if "MEAN_B02" not in df.columns:
        raise ValueError("Could not find MEAN_B02; cannot infer remote columns start.")
    start_idx = list(df.columns).index("MEAN_B02")
    remote_cols = list(df.columns)[start_idx:]

    return id_cols, found_targets, ignore_cols, remote_cols


def to_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn


def correlation_prune(X: pd.DataFrame, threshold: float = 0.98, log_file=None) -> list[str]:
    """
    Remove near-duplicate predictors by absolute correlation threshold.
    """
    Xv = X.copy()
    # drop constant columns
    nunique = Xv.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    Xv = Xv[keep]

    if Xv.shape[1] < 2:
        return list(Xv.columns)

    corr = Xv.corr().abs()
    cols = list(corr.columns)
    to_drop = set()

    t0 = time.perf_counter()
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.iloc[i, j] >= threshold:
                to_drop.add(cols[j])

    log(f"Correlation check done in {fmt_seconds(time.perf_counter() - t0)}. Dropping {len(to_drop)} columns.",
        log_file)
    kept = [c for c in Xv.columns if c not in to_drop]
    return kept


def filter_by_index_metadata(remote_cols: list[str], xlsx_path: Path, domain: str = "moisture") -> list[str]:
    """
    Uses bazaIndeksowRS_2T_1.xlsx-like metadata to select indices by application_domain.
    Always keeps Sentinel-2 band means (MEAN_Bxx).
    """
    meta = pd.read_excel(xlsx_path)
    meta.columns = [c.strip() for c in meta.columns]
    if "skrot" not in meta.columns or "application_domain" not in meta.columns:
        raise ValueError("Index metadata xlsx must contain columns: 'skrot', 'application_domain'.")

    domain_mask = meta["application_domain"].astype(str).str.contains(domain, case=False, na=False)
    allowed = set(meta.loc[domain_mask, "skrot"].astype(str).str.strip().tolist())

    kept = []
    for c in remote_cols:
        if c.startswith("MEAN_B"):
            kept.append(c)
        elif c.strip() in allowed:
            kept.append(c)

    if len(kept) < 5:
        warnings.warn(f"Domain filter kept only {len(kept)} columns. Check metadata domain='{domain}'.")
    return kept


def make_group_cv(groups: np.ndarray, max_splits: int = 10, seed: int = 42):
    ug = np.unique(groups)
    # If using 'Site_Part' (L, Z, LMU...) we have ~7 groups.
    # GroupKFold requires n_splits <= n_groups.
    if len(ug) >= 3:
        return GroupKFold(n_splits=min(max_splits, len(ug)))
    # fallback if too few groups
    return KFold(n_splits=min(5, len(groups)), shuffle=True, random_state=seed)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Handle edge case of constant prediction or single sample
    if len(y_true) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "SpearmanR": np.nan}

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        rho = float(spearmanr(y_true, y_pred).correlation)
    except:
        rho = np.nan
    return {"R2": float(r2), "RMSE": rmse, "MAE": mae, "SpearmanR": rho}


# --------------------------- MODEL BUILDING ----------------------------------

def build_models(seed: int = 42, use_stacking: bool = False):
    """
    Returns dict: name -> (estimator_pipeline, param_grid)
    Enhanced with PowerTransformer and Feature Selection (SelectKBest).
    """
    # Base Steps: Impute -> Scale -> Select Top 20 Features (dynamic per fold)
    # PowerTransformer helps normalize non-gaussian targets (like ka010 or GWL)
    base_numeric = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k=20))
    ]

    models = {}

    # 1. Ridge
    ridge = Pipeline(base_numeric + [("model", Ridge(random_state=seed))])
    ridge_grid = {"model__alpha": np.logspace(-4, 4, 25)}
    models["Ridge"] = (ridge, ridge_grid)

    # 2. Elastic Net
    enet = Pipeline(base_numeric + [("model", ElasticNet(max_iter=20000, random_state=seed))])
    enet_grid = {
        "model__alpha": np.logspace(-4, 2, 20),
        "model__l1_ratio": [0.1, 0.5, 0.9],
    }
    models["ElasticNet"] = (enet, enet_grid)

    # 3. PLSR (Partial Least Squares handles collinearity natively)
    # Note: PLSR doesn't need SelectKBest strictly, but we keep it uniform
    plsr = Pipeline(base_numeric + [("model", PLSRegression())])
    plsr_grid = {"model__n_components": list(range(2, 12))}
    models["PLSR"] = (plsr, plsr_grid)

    # 4. SVR (RBF)
    svr = Pipeline(base_numeric + [("model", SVR(kernel="rbf"))])
    svr_grid = {
        "model__C": [1, 10, 50, 100],
        "model__gamma": ["scale", 0.01, 0.1],
        "model__epsilon": [0.01, 0.1, 0.2],
    }
    models["SVR_RBF"] = (svr, svr_grid)

    # 5. Gaussian Process (GPR)
    # GPR is sensitive to feature count; k=15 or k=20 helps significantly
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gpr = Pipeline(
        base_numeric + [("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed))])
    models["GPR"] = (gpr, {})

    # 6. Random Forest
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # RF performs internal feature selection, but removing noise helps on small N
        ("selector", SelectKBest(score_func=f_regression, k=25)),
        ("model", RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1))
    ])
    rf_grid = {
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.3, 0.5],
    }
    models["RandomForest"] = (rf, rf_grid)

    # 7. HistGradientBoosting
    hgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(random_state=seed))
    ])
    hgb_grid = {
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__l2_regularization": [0.0, 0.1],
    }
    models["HistGB"] = (hgb, hgb_grid)

    # 8. Stacking Regressor (Optional)
    if use_stacking:
        # Define Level-0 Estimators (Diverse set)
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=seed)),
            ('svr', SVR(kernel='rbf', C=10, epsilon=0.1)),
            ('ridge', Ridge(alpha=1.0))
        ]
        # Level-1 Estimator (Meta-model)
        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),  # Meta-learner
            passthrough=False
        )
        stack_pipe = Pipeline(base_numeric + [('model', stack)])
        models["Stacking"] = (stack_pipe, {})

    return models


# --------------------------- EVALUATION --------------------------------------

def nested_group_cv_evaluate(
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        models: dict,
        seed: int,
        outer_name: str,
        outdir: Path,
        log_file: Path
):
    log(f"--- Starting Evaluation: {outer_name} ---", log_file)
    t_start = time.perf_counter()

    out_rows = []
    pred_rows = []

    # Use PowerTransformer to normalize y (soil moisture/GWL is often skewed)
    pt = PowerTransformer(method='yeo-johnson')
    y_trans = pt.fit_transform(y.reshape(-1, 1)).ravel()

    outer_cv = make_group_cv(groups, max_splits=50, seed=seed)

    # Pre-calculate folds to ensure consistency across models
    splits = list(outer_cv.split(X, y_trans, groups=groups))
    n_folds = len(splits)
    log(f"Outer CV has {n_folds} folds.", log_file)

    for model_name, (pipe, grid) in models.items():
        t_model = time.perf_counter()
        log(f"Evaluating {model_name}...", log_file)

        fold = 0
        for train_idx, test_idx in splits:
            fold += 1
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y_trans[train_idx], y_trans[test_idx]
            gtr = groups[train_idx]

            # Inner CV for Hyperparameter Tuning
            inner_cv = make_group_cv(gtr, max_splits=5, seed=seed)

            if grid:
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    cv=inner_cv.split(Xtr, ytr, groups=gtr) if isinstance(inner_cv, GroupKFold) else inner_cv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(Xtr, ytr)
                best = search.best_estimator_
                best_params = search.best_params_
            else:
                best = pipe
                best.fit(Xtr, ytr)
                best_params = {}

            # Predict in transformed space
            yhat_trans = best.predict(Xte).ravel()

            # Inverse transform to get real units (cm or %)
            yhat_real = pt.inverse_transform(yhat_trans.reshape(-1, 1)).ravel()
            yte_real = pt.inverse_transform(yte.reshape(-1, 1)).ravel()

            m = metrics(yte_real, yhat_real)

            out_rows.append(
                {
                    "outer_cv": outer_name,
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **m,
                    "best_params": str(best_params),
                }
            )

            for i, idx in enumerate(test_idx):
                pred_rows.append(
                    {
                        "outer_cv": outer_name,
                        "model": model_name,
                        "fold": fold,
                        "row_index": int(idx),
                        "group": str(groups[idx]),
                        "y_true": float(yte_real[i]),
                        "y_pred": float(yhat_real[i]),
                    }
                )

        elapsed = time.perf_counter() - t_model
        log(f"  > {model_name} finished in {fmt_seconds(elapsed)}", log_file)

    df_folds = pd.DataFrame(out_rows)
    df_preds = pd.DataFrame(pred_rows)

    # Summary Statistics
    summary = (
        df_folds.groupby(["outer_cv", "model"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            SpearmanR_mean=("SpearmanR", "mean"),
        )
        .sort_values(["RMSE_mean"])
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(outdir / f"fold_metrics_{outer_name}.csv", index=False)
    df_preds.to_csv(outdir / f"predictions_{outer_name}.csv", index=False)
    summary.to_csv(outdir / f"summary_{outer_name}.csv", index=False)

    log(f"{outer_name} complete in {fmt_seconds(time.perf_counter() - t_start)}", log_file)
    return df_folds, summary


def fit_final_and_importance(X: pd.DataFrame, y: np.ndarray, model_pipe: Pipeline, outpath: Path, log_file,
                             seed: int = 42):
    log("Fitting final model for importance...", log_file)
    model_pipe.fit(X, y)

    log("Computing permutation importance (50 repeats)...", log_file)
    r = permutation_importance(
        model_pipe,
        X,
        y,
        n_repeats=50,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp.to_csv(outpath, index=False)
    log(f"Importance saved to {outpath.name}", log_file)
    return imp


# --------------------------- MAIN --------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_CSV), help="Path to csv.")
    parser.add_argument("--index_meta_xlsx", type=str, default=str(DEFAULT_XLSX), help="Path to xlsx.")
    parser.add_argument("--domain_filter", type=str, default="", help="Filter by domain (e.g. moisture).")
    parser.add_argument("--target", type=str, default="ka010", help="Target column (e.g. ka010, GWL, NDVI).")
    parser.add_argument("--corr_prune", type=float, default=0.95,
                        help="Correlation prune threshold (0.95 recommended).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Output directory.")
    parser.add_argument("--do_importance", action="store_true", help="Compute variable importance.")
    parser.add_argument("--best_model_for_importance", type=str, default="RandomForest", help="Model to explain.")
    parser.add_argument("--stacking", action="store_true", help="Include Stacking Regressor.")
    parser.add_argument("--mix_sites", action="store_true", default=True, help="Treat all sites as one pool (Random CV).")

    args = parser.parse_args()

    csv_path = Path(args.data_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_file = outdir / "run_log.txt"
    # clear old log
    if log_file.exists():
        log_file.unlink()

    log("--- Script Start ---", log_file)
    log(f"Target Variable: {args.target}", log_file)
    log(f"Data Path: {csv_path}", log_file)

    t_global = time.perf_counter()

    # 1. Load Data
    df = read_dataset(csv_path)
    id_cols, targets, ignore_cols, remote_cols = infer_columns(df)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in columns. Available: {targets}")

    # predictors: remote only
    X = df[remote_cols].copy()

    # optional index metadata filtering
    if args.domain_filter:
        kept = filter_by_index_metadata(remote_cols, Path(args.index_meta_xlsx), domain=args.domain_filter)
        X = df[kept].copy()
        log(f"Domain filter '{args.domain_filter}' reduced features to {X.shape[1]}", log_file)

    X = to_numeric_frame(X)
    y = pd.to_numeric(df[args.target], errors="coerce").to_numpy()

    # drop rows with missing y
    ok = ~np.isnan(y)
    X = X.loc[ok].reset_index(drop=True)
    y = y[ok]
    df2 = df.loc[ok].reset_index(drop=True)

    log(f"Valid samples (N): {len(y)}", log_file)

    # 2. Prune highly correlated predictors
    log(f"Pruning features with corr > {args.corr_prune}...", log_file)
    kept_cols = correlation_prune(X, threshold=args.corr_prune, log_file=log_file)
    X = X[kept_cols]

    # 3. Define Groups
    # 'Dayn' for LODO
    groups_day = df2["Dayn"].astype(str).to_numpy()

    # 'SiteTerm' logic:
    if args.mix_sites:
        log("Validation Strategy: Random K-Fold (Mixing all sites)", log_file)
        # Create a unique ID for every row.
        # make_group_cv falls back to KFold if n_groups == n_samples
        groups_site = np.arange(len(df2))
    else:
        log("Validation Strategy: Leave-One-Site-Out (Spatial Grouping)", log_file)
        site_raw = df2["SiteTerm"].astype(str)
        groups_site = site_raw.apply(lambda s: s.split("_")[0] if "_" in s else s).to_numpy()
        unique_sites = np.unique(groups_site)
        log(f"Unique Sites found for grouping: {len(unique_sites)} -> {unique_sites}", log_file)

    # 4. Build Models
    models = build_models(seed=args.seed, use_stacking=args.stacking)

    # 5. Evaluate Leave-One-Day-Out
    nested_group_cv_evaluate(
        X=X, y=y, groups=groups_day, models=models, seed=args.seed,
        outer_name="leave_one_day_out", outdir=outdir, log_file=log_file
    )

    # 6. Evaluate Leave-One-Site-Out (Or Random CV if mixed)
    nested_group_cv_evaluate(
        X=X, y=y, groups=groups_site, models=models, seed=args.seed,
        outer_name="leave_one_site_out" if not args.mix_sites else "random_cv_mixed",
        outdir=outdir, log_file=log_file
    )

    # 7. Importance (Optional)
    if args.do_importance:
        if args.best_model_for_importance not in models:
            warnings.warn(f"Model '{args.best_model_for_importance}' not found. Skipping importance.")
        else:
            log(f"Computing Importance for {args.best_model_for_importance}...", log_file)
            pipe, grid = models[args.best_model_for_importance]

            # Simple tune on full data
            if grid:
                cv = make_group_cv(groups_day, max_splits=5, seed=args.seed)
                search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
                search.fit(X, y)
                final_model = search.best_estimator_
            else:
                final_model = pipe

            imp_path = outdir / f"importance_{args.target}_{args.best_model_for_importance}.csv"
            fit_final_and_importance(X, y, final_model, imp_path, log_file, seed=args.seed)

    total_time = time.perf_counter() - t_global
    log(f"DONE. Total execution time: {fmt_seconds(total_time)}", log_file)
    log(f"Outputs written to: {outdir.resolve()}", log_file)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()