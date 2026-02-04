#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Soil moisture modeling from satellite indices (data-driven benchmark suite)

Target:
  - default: ka010 (TDR dielectric constant, 0–10 cm)

Predictors:
  - remote predictors from MEAN_B02 .. last column (as in your combined CSV)

Validation:
  - leave-one-day-out (group by Dayn)
  - leave-one-site-out (group by SiteTerm)

Models:
  - Ridge, ElasticNet, PLSR, SVR(RBF), Gaussian Process, Random Forest, HistGradientBoosting

Outputs:
  - fold_metrics_*.csv
  - predictions_*.csv
  - summary_*.csv
  - optional permutation_importance_*.csv
  - run_log.txt (timing log)

Requirements:
  pandas numpy scikit-learn scipy openpyxl
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
    import site  # stdlib
    site.ENABLE_USER_SITE = False

    user_site = site.getusersitepackages()
    user_sites = [user_site] if isinstance(user_site, str) else list(user_site)

    def _is_user_site(p: str) -> bool:
        ap = os.path.abspath(p).lower()
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


# --------------------------- DEFAULT PATHS (edit if needed) -------------------
BASE_DIR = Path(r"E:\pythonProject\Soil_project")
DEFAULT_CSV = BASE_DIR / "zeszyt1.csv"
DEFAULT_XLSX = BASE_DIR / "bazaIndeksowRS_2T_1.xlsx"
DEFAULT_OUTDIR = BASE_DIR / "outputs"
# -----------------------------------------------------------------------------


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    sec = s - 60 * m
    if m < 60:
        return f"{m}m {sec:.0f}s"
    h = int(m // 60)
    m2 = m - 60 * h
    return f"{h}h {m2}m {sec:.0f}s"


def log(msg: str, log_file: Path | None = None):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_columns(df: pd.DataFrame):
    id_cols = ["Dayn", "SiteTerm"]
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    field_cols = ["ka010", "eps010", "HeSpec", "GWL", "NDVI"]
    for c in field_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required field column: {c}")

    ignore_candidates = ["rb", "pd", "Po", "alfa", "teta1", "teta2"]
    ignore_cols = [c for c in ignore_candidates if c in df.columns]

    if "MEAN_B02" not in df.columns:
        raise ValueError("Could not find MEAN_B02; cannot infer remote columns start.")
    start_idx = list(df.columns).index("MEAN_B02")
    remote_cols = list(df.columns)[start_idx:]

    return id_cols, field_cols, ignore_cols, remote_cols


def to_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    return Xn


def correlation_prune(X: pd.DataFrame, threshold: float = 0.98) -> list[str]:
    """
    Remove near-duplicate predictors by absolute correlation threshold.
    Keeps the first feature encountered and drops later ones that are too correlated.
    """
    Xv = X.copy()

    nunique = Xv.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    Xv = Xv[keep]

    if Xv.shape[1] <= 2:
        return list(Xv.columns)

    corr = Xv.corr().abs()
    cols = list(corr.columns)
    to_drop = set()
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.iloc[i, j] >= threshold:
                to_drop.add(cols[j])

    kept = [c for c in Xv.columns if c not in to_drop]
    return kept


def filter_by_index_metadata(remote_cols: list[str], xlsx_path: Path, domain: str) -> list[str]:
    """
    Select indices by application_domain from bazaIndeksowRS_2T_1.xlsx.
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

    return kept


def make_group_cv(groups: np.ndarray, max_splits: int = 10, seed: int = 42):
    ug = np.unique(groups)
    if len(ug) >= 3:
        return GroupKFold(n_splits=min(max_splits, len(ug)))
    return KFold(n_splits=min(5, len(groups)), shuffle=True, random_state=seed)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rho = float(spearmanr(y_true, y_pred).correlation) if len(y_true) > 2 else np.nan
    return {"R2": float(r2), "RMSE": rmse, "MAE": mae, "SpearmanR": rho}


def build_models(seed: int = 42):
    base_numeric = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    models: dict[str, tuple[Pipeline, dict]] = {}

    ridge = Pipeline(base_numeric + [("model", Ridge(random_state=seed))])
    models["Ridge"] = (ridge, {"model__alpha": np.logspace(-4, 4, 25)})

    enet = Pipeline(base_numeric + [("model", ElasticNet(max_iter=20000, random_state=seed))])
    models["ElasticNet"] = (
        enet,
        {"model__alpha": np.logspace(-4, 2, 20), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    )

    plsr = Pipeline(base_numeric + [("model", PLSRegression())])
    models["PLSR"] = (plsr, {"model__n_components": list(range(2, 16))})

    svr = Pipeline(base_numeric + [("model", SVR(kernel="rbf"))])
    models["SVR_RBF"] = (
        svr,
        {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.1, 0.01], "model__epsilon": [0.01, 0.05, 0.1, 0.2]},
    )

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    gpr = Pipeline(base_numeric + [("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed))])
    models["GPR"] = (gpr, {})

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=800, random_state=seed, n_jobs=-1)),
        ]
    )
    models["RandomForest"] = (
        rf,
        {"model__max_depth": [None, 5, 10, 15], "model__min_samples_leaf": [1, 3, 5], "model__max_features": ["sqrt", 0.5, 0.8]},
    )

    hgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=seed)),
        ]
    )
    models["HistGB"] = (
        hgb,
        {"model__max_depth": [3, 5, 7], "model__learning_rate": [0.03, 0.06, 0.1], "model__l2_regularization": [0.0, 0.1, 1.0], "model__max_leaf_nodes": [15, 31, 63]},
    )

    return models


def nested_group_cv_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict,
    seed: int,
    outer_name: str,
    outdir: Path,
    log_file: Path | None,
):
    out_rows = []
    pred_rows = []

    outer_cv = make_group_cv(groups, max_splits=50, seed=seed)
    n_folds = outer_cv.get_n_splits(X, y, groups)

    log(f"CV start: {outer_name} | folds={n_folds} | n={len(y)} | p={X.shape[1]}", log_file)

    t_outer0 = time.perf_counter()

    for model_name, (pipe, grid) in models.items():
        t_model0 = time.perf_counter()
        log(f"  Model start: {model_name} | grid={'yes' if bool(grid) else 'no'}", log_file)

        fold = 0
        for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
            fold += 1
            t_fold0 = time.perf_counter()

            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            gtr = groups[train_idx]

            inner_cv = make_group_cv(gtr, max_splits=5, seed=seed)

            t_fit0 = time.perf_counter()
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

            fit_time = time.perf_counter() - t_fit0

            yhat = best.predict(Xte).ravel()
            m = metrics(yte, yhat)

            fold_time = time.perf_counter() - t_fold0
            log(
                f"    Fold {fold}/{n_folds} done | model={model_name} | fit={fmt_seconds(fit_time)} | "
                f"fold_total={fmt_seconds(fold_time)} | RMSE={m['RMSE']:.4g} | R2={m['R2']:.3f}",
                log_file,
            )

            out_rows.append(
                {
                    "outer_cv": outer_name,
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **m,
                    "best_params": str(best_params),
                    "fit_seconds": float(fit_time),
                    "fold_seconds": float(fold_time),
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
                        "y_true": float(y[idx]),
                        "y_pred": float(yhat[i]),
                    }
                )

        model_time = time.perf_counter() - t_model0
        log(f"  Model end: {model_name} | elapsed={fmt_seconds(model_time)}", log_file)

    df_folds = pd.DataFrame(out_rows)
    df_preds = pd.DataFrame(pred_rows)

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
            SpearmanR_std=("SpearmanR", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fold_seconds_mean=("fold_seconds", "mean"),
        )
        .sort_values(["outer_cv", "RMSE_mean"])
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(outdir / f"fold_metrics_{outer_name}.csv", index=False)
    df_preds.to_csv(outdir / f"predictions_{outer_name}.csv", index=False)
    summary.to_csv(outdir / f"summary_{outer_name}.csv", index=False)

    outer_time = time.perf_counter() - t_outer0
    log(f"CV end: {outer_name} | elapsed={fmt_seconds(outer_time)}", log_file)

    return df_folds, summary


def fit_final_and_importance(X: pd.DataFrame, y: np.ndarray, model_pipe: Pipeline, outpath: Path, log_file: Path | None, seed: int = 42):
    t0 = time.perf_counter()
    log("Permutation importance: fit model on full dataset...", log_file)
    model_pipe.fit(X, y)
    log("Permutation importance: computing...", log_file)

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
        {"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(outpath, index=False)

    log(f"Permutation importance saved: {outpath} | elapsed={fmt_seconds(time.perf_counter() - t0)}", log_file)
    return imp


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_CSV),
                        help="Path to zeszyt1.csv (semicolon-separated).")
    parser.add_argument("--index_meta_xlsx", type=str, default=str(DEFAULT_XLSX),
                        help="Path to bazaIndeksowRS_2T_1.xlsx.")
    parser.add_argument("--domain_filter", type=str, default="",
                        help="If set (e.g., moisture), filters indices using metadata; if empty uses all remote predictors.")
    parser.add_argument("--target", type=str, default="ka010",
                        help="Target variable (default: ka010).")
    parser.add_argument("--corr_prune", type=float, default=0.98,
                        help="Correlation prune threshold (abs r).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR),
                        help="Output directory.")
    parser.add_argument("--do_importance", action="store_true",
                        help="Compute permutation importance for one chosen model (fit on full data).")
    parser.add_argument("--best_model_for_importance", type=str, default="ElasticNet",
                        help="ElasticNet, Ridge, PLSR, SVR_RBF, GPR, RandomForest, HistGB")
    parser.add_argument("--debug_paths", action="store_true",
                        help="Print Python and sklearn paths (useful in PyCharm).")

    args = parser.parse_args()

    t_total0 = time.perf_counter()

    csv_path = Path(args.data_csv)
    xlsx_path = Path(args.index_meta_xlsx)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / "run_log.txt"

    # Start log
    log("Run started.", log_file)
    log(f"Python executable: {sys.executable}", log_file)

    if args.debug_paths:
        import sklearn
        log(f"sklearn version: {sklearn.__version__}", log_file)
        log(f"sklearn path   : {sklearn.__file__}", log_file)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log(f"Reading CSV: {csv_path}", log_file)
    t0 = time.perf_counter()
    df = read_dataset(csv_path)
    log(f"CSV loaded | rows={df.shape[0]} cols={df.shape[1]} | elapsed={fmt_seconds(time.perf_counter() - t0)}", log_file)

    log("Inferring columns...", log_file)
    t0 = time.perf_counter()
    _, _, _, remote_cols = infer_columns(df)
    log(f"Remote predictors inferred | count={len(remote_cols)} | elapsed={fmt_seconds(time.perf_counter() - t0)}", log_file)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not in columns. Available: {df.columns.tolist()}")

    # predictors: remote only
    log("Preparing predictors (remote only)...", log_file)
    t0 = time.perf_counter()
    X = df[remote_cols].copy()

    # optional metadata filter
    if args.domain_filter:
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Index metadata xlsx not found: {xlsx_path}")
        log(f"Applying metadata domain filter: {args.domain_filter}", log_file)
        kept = filter_by_index_metadata(remote_cols, xlsx_path, domain=args.domain_filter)
        X = df[kept].copy()
        log(f"Domain filter kept predictors | count={len(kept)}", log_file)

    X = to_numeric_frame(X)
    y = pd.to_numeric(df[args.target], errors="coerce").to_numpy()

    ok = ~np.isnan(y)
    X = X.loc[ok].reset_index(drop=True)
    y = y[ok]
    df2 = df.loc[ok].reset_index(drop=True)
    log(f"Prepared X,y | n={len(y)} p={X.shape[1]} | elapsed={fmt_seconds(time.perf_counter() - t0)}", log_file)

    # prune highly correlated predictors
    log(f"Correlation pruning | threshold={args.corr_prune}", log_file)
    t0 = time.perf_counter()
    kept_cols = correlation_prune(X, threshold=args.corr_prune)
    X = X[kept_cols]
    log(f"Pruning done | kept_p={X.shape[1]} | elapsed={fmt_seconds(time.perf_counter() - t0)}", log_file)

    # groups
    groups_day = df2["Dayn"].astype(str).to_numpy()
    groups_site = df2["SiteTerm"].astype(str).to_numpy()
    log(f"Groups | unique Dayn={len(np.unique(groups_day))} | unique SiteTerm={len(np.unique(groups_site))}", log_file)

    models = build_models(seed=args.seed)
    log(f"Models prepared: {', '.join(models.keys())}", log_file)

    # evaluate leave-one-day-out
    nested_group_cv_evaluate(
        X=X, y=y, groups=groups_day, models=models, seed=args.seed,
        outer_name="leave_one_day_out", outdir=outdir, log_file=log_file
    )

    # evaluate leave-one-site-out
    nested_group_cv_evaluate(
        X=X, y=y, groups=groups_site, models=models, seed=args.seed,
        outer_name="leave_one_site_out", outdir=outdir, log_file=log_file
    )

    # optional permutation importance
    if args.do_importance:
        if args.best_model_for_importance not in models:
            raise ValueError(f"Unknown model '{args.best_model_for_importance}'. Options: {list(models.keys())}")

        pipe, grid = models[args.best_model_for_importance]
        log(f"Final importance model: {args.best_model_for_importance}", log_file)

        if grid:
            cv = make_group_cv(groups_day, max_splits=5, seed=args.seed)
            log("Final importance: grid search on full data (not a generalization estimate)...", log_file)
            t0 = time.perf_counter()
            search = GridSearchCV(
                pipe,
                grid,
                cv=cv.split(X, y, groups=groups_day) if isinstance(cv, GroupKFold) else cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X, y)
            final_model = search.best_estimator_
            log(f"Final grid search done | elapsed={fmt_seconds(time.perf_counter() - t0)} | best={search.best_params_}", log_file)
        else:
            final_model = pipe

        imp_path = outdir / f"permutation_importance_{args.best_model_for_importance}.csv"
        fit_final_and_importance(X, y, final_model, imp_path, log_file, seed=args.seed)

    log(f"Run finished | total elapsed={fmt_seconds(time.perf_counter() - t_total0)}", log_file)
    print(f"Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
