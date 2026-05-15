# -*- coding: utf-8 -*-
"""
Heat Acclimation — Skin temperature drift pipeline (neck + ankle + DPG), aligned to CBT assumptions.

NEW in this version:
- Computes AIC/BIC model-selection outputs for EACH signal (neck/ankle/DPG) across 3 formulas:
    M0: drift ~ arm * scenario
    M1: drift ~ arm * scenario + sex_c
    M2: drift ~ arm * scenario + sex_c + fat_pct
  with (1|part_id) in all cases.
- Exports a SINGLE combined CSV:
    SkinTemp_model_selection_AIC_BIC.csv
  (site × model rows)

Important:
- AIC/BIC are only meaningful under ML (REML=False), so these selection fits use reml=False.
- Your main inference model can still be fit with reml=True (unchanged) if you prefer; this code keeps your
  existing final-model fit as-is, but uses ML specifically for AIC/BIC reporting.
"""

from __future__ import annotations

from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure editable text in vector exports (Illustrator-friendly)
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import t as t_dist
from scipy import stats



# =========================================================
# REPOSITORY PATHS
# =========================================================
# This file is intended to live in: <repo>/code/03_analysis/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_METADATA = REPO_ROOT / "data" / "metadata"
OUTPUTS = REPO_ROOT / "outputs" / "model_outputs" / "skin_temp_outputs"

# =========================================================
# CONFIG
# =========================================================

PATH_SKIN = DATA_PROCESSED / "skin_temp.csv"
PATH_PART = DATA_METADATA / "participant_meta_pseud.csv"

OUTDIR = OUTPUTS / "skin_temperature" / "primary_drift"
OUTDIR.mkdir(parents=True, exist_ok=True)

REQ_SKIN = [
    "datetime", "part_id", "condition", "scenario_short", "session_id",
    "skin_temp_neck", "skin_temp_ankle",
    "qc_neck_sudden_drop", "qc_ankle_sudden_drop",
]
REQ_PART_BASE = ["part_id", "sex"]  # fat_pct is optional but used if present

VALID_ARMS = ["FR", "CC"]
ARM_ORDER = ["FR", "CC"]

VALID_SCENARIO_SHORT = ["HS_pre", "HS_post"]
SCENARIO_MAP = {"HS_pre": "HS1", "HS_post": "HS2"}
SCEN_ORDER = ["HS1", "HS2"]

# Drift windows (kept as in your latest code paste)
EARLY_START = "12:15"
EARLY_END   = "12:30"
LATE_START  = "16:15"
LATE_END    = "16:30"

MIN_VALID_MINUTES_PER_WINDOW = 10
RESAMPLE_TO_MINUTE = True

Z_OUTLIER_THRESH_SITE = 3.0
TREAT_NA_QC_AS_OK = True

DROP_ASYMMETRICAL_PARTICIPANTS = True
REQUIRED_SCENARIOS_PER_PART = 2

PAIR_LINES = True
JITTER = 0.10
POINT_SIZE = 46
ERR_LW = 2.2
AXH0_LW = 1.2
LABEL_PARTICIPANTS = True
LABEL_FONTSIZE = 7.5
SAVE_PNG = True
SAVE_PDF = True


# =========================================================
# PRE-MODEL FILTER RULES (MATCH CBT)
# =========================================================

EXCLUDE_PARTICIPANTS_ALL = {"P05"}

EXCLUDE_PARTICIPANTS_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): {"P02"},
}

SESSION_START_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): "10:40",
    pd.Timestamp("2023-06-06").normalize(): "09:55",
    pd.Timestamp("2023-06-07").normalize(): "09:40",
}

SESSION_START_BY_DATE_PART = {
    (pd.Timestamp("2023-06-22").normalize(), "P06"): "10:30",
}

DEFAULT_SESSION_END = "16:30"


# =========================================================
# HELPERS
# =========================================================

def require_columns(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}\nAvailable: {list(df.columns)}")

def ensure_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clock_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()
    df["date"] = df["datetime"].dt.normalize()
    df["minute_floor"] = df["datetime"].dt.floor("min")
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    return df

def in_window(mins: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    a = clock_to_minutes(start_hhmm)
    b = clock_to_minutes(end_hhmm)
    return (mins >= a) & (mins < b)

def robust_ci_from_t(mean, se, df_resid, alpha=0.05):
    if not np.isfinite(se) or se <= 0 or not np.isfinite(df_resid) or df_resid <= 0:
        return np.nan, np.nan
    tcrit = float(t_dist.ppf(1 - alpha / 2, df_resid))
    return mean - tcrit * se, mean + tcrit * se

def normalize_bool_series(x: pd.Series, na_is_ok: bool = True) -> pd.Series:
    s = x.copy()
    if s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        is_bad = ss.isin(["true", "1", "yes", "y", "bad"])
        is_ok  = ss.isin(["false", "0", "no", "n", "ok"])
        out = pd.Series(np.where(is_bad, True, np.where(is_ok, False, np.nan)), index=s.index)
    else:
        sn = pd.to_numeric(s, errors="coerce")
        out = (sn == 1)
        out = out.astype(float)

    if isinstance(out.dtype, pd.BooleanDtype) or out.dtype == bool:
        out = out.astype("float")

    if na_is_ok:
        out = out.fillna(0.0)
    else:
        out = out.fillna(1.0)

    return out.astype(bool)


# =========================================================
# SESSION START OVERRIDE FILTER
# =========================================================

def _get_start_hhmm_for_row(date_val: pd.Timestamp, part_id: str) -> str:
    key = (pd.Timestamp(date_val).normalize(), str(part_id))
    if key in SESSION_START_BY_DATE_PART:
        return SESSION_START_BY_DATE_PART[key]
    d = pd.Timestamp(date_val).normalize()
    return SESSION_START_BY_DATE.get(d, "09:30")

def apply_exclusions_and_session_clipping(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df = add_time_fields(df)

    log_rows = []

    if EXCLUDE_PARTICIPANTS_ALL:
        m = df["part_id"].astype(str).isin(list(EXCLUDE_PARTICIPANTS_ALL))
        if m.any():
            for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
                log_rows.append({"action": "exclude_all", "part_id": pid, "date": pd.NaT, "reason": "dropped_out"})
        df = df.loc[~m].copy()

    for d, pids in EXCLUDE_PARTICIPANTS_BY_DATE.items():
        m = (df["date"] == pd.Timestamp(d).normalize()) & (df["part_id"].astype(str).isin(list(pids)))
        if m.any():
            for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
                log_rows.append({"action": "exclude_date", "part_id": pid, "date": pd.Timestamp(d).normalize(), "reason": "manual_exclusion"})
        df = df.loc[~m].copy()

    def row_keep(r):
        d = r["date"]
        pid = str(r["part_id"])
        start_hhmm = _get_start_hhmm_for_row(d, pid)
        start_m = clock_to_minutes(start_hhmm)
        end_m   = clock_to_minutes(DEFAULT_SESSION_END)
        return (r["minute_of_day"] >= start_m) and (r["minute_of_day"] <= end_m)

    keep_mask = df.apply(row_keep, axis=1)
    clipped = df.loc[~keep_mask].copy()

    if not clipped.empty:
        g = clipped.groupby(["date", "part_id"]).size().reset_index(name="n_rows_clipped")
        for _, r in g.iterrows():
            log_rows.append({
                "action": "clip_before_start",
                "part_id": str(r["part_id"]),
                "date": pd.Timestamp(r["date"]).normalize(),
                "reason": f"start_override={_get_start_hhmm_for_row(pd.Timestamp(r['date']).normalize(), str(r['part_id']))}",
                "n_rows_clipped": int(r["n_rows_clipped"]),
            })

    df = df.loc[keep_mask].copy()

    log_cols = ["action", "part_id", "date", "reason", "n_rows_clipped"]
    log = pd.DataFrame(log_rows, columns=log_cols)
    return df, log


# =========================================================
# MIXEDLM CONTRAST UTILITIES
# =========================================================

def build_design_row_from_res(res, new_df: pd.DataFrame) -> np.ndarray:
    X = dmatrix(res.model.data.design_info, new_df, return_type="dataframe")
    return X.iloc[0].values

def estimate_fe_linear_combo(res, cvec: np.ndarray):
    beta = np.asarray(res.fe_params).reshape(-1, 1)
    k = beta.shape[0]

    Vfull = np.asarray(res.cov_params())
    V = Vfull[:k, :k]

    c = np.asarray(cvec).reshape(-1, 1)
    if c.shape[0] != k:
        raise ValueError(f"Contrast length {c.shape[0]} does not match #fixed effects {k}")

    mean = float((c.T @ beta)[0, 0])
    var  = float((c.T @ V @ c)[0, 0])
    se   = float(np.sqrt(max(var, 0.0)))

    df_resid = float(getattr(res, "df_resid", np.nan))
    if np.isfinite(df_resid) and df_resid > 0 and se > 0:
        tval = mean / se
        p = 2 * (1 - t_dist.cdf(abs(tval), df_resid))
        lo, hi = robust_ci_from_t(mean, se, df_resid)
    else:
        tval = mean / se if se > 0 else np.nan
        p = np.nan
        lo, hi = mean - 1.96 * se, mean + 1.96 * se

    return mean, se, tval, p, lo, hi


# =========================================================
# LOAD / PREP
# =========================================================

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    skin = pd.read_csv(PATH_SKIN)
    skin.columns = [c.strip() for c in skin.columns]
    require_columns(skin, REQ_SKIN, "SkinTemp")

    skin["part_id"] = skin["part_id"].astype(str)
    skin["condition"] = skin["condition"].astype(str)
    skin["scenario_short"] = skin["scenario_short"].astype(str)
    skin = ensure_numeric(skin, ["skin_temp_neck", "skin_temp_ankle"])

    skin = skin[skin["condition"].isin(VALID_ARMS)].copy()
    skin = skin[skin["scenario_short"].isin(VALID_SCENARIO_SHORT)].copy()
    skin["arm"] = skin["condition"]
    skin["scenario"] = skin["scenario_short"].map(SCENARIO_MAP)

    skin, filt_log = apply_exclusions_and_session_clipping(skin)

    # participant meta
    part = pd.read_csv(PATH_PART)
    part.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in part.columns]
    require_columns(part, REQ_PART_BASE, "Participants")
    part["part_id"] = part["part_id"].astype(str)

    # fat_pct is optional but we pull it if present
    if "fat_pct" in part.columns:
        part["fat_pct"] = pd.to_numeric(part["fat_pct"], errors="coerce")

    part = part.copy()
    part["sex"] = part["sex"].astype(str).str.strip().str.upper()
    part["sex_c"] = part["sex"].map({"F": 1, "M": 0, "FEMALE": 1, "MALE": 0}).astype(float)

    keep_cols = ["part_id", "sex_c"]
    if "fat_pct" in part.columns:
        keep_cols.append("fat_pct")

    df = skin.merge(part[keep_cols], on="part_id", how="left")

    return df, part, filt_log


def resample_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    d = add_time_fields(df.copy())

    if not RESAMPLE_TO_MINUTE:
        d["dpg"] = d["skin_temp_ankle"] - d["skin_temp_neck"]
        return d

    gcols = ["part_id", "arm", "scenario", "session_id", "minute_floor"]
    agg = d.groupby(gcols, as_index=False).agg(
        skin_temp_neck=("skin_temp_neck", "mean"),
        skin_temp_ankle=("skin_temp_ankle", "mean"),
        qc_neck_sudden_drop=("qc_neck_sudden_drop", "first"),
        qc_ankle_sudden_drop=("qc_ankle_sudden_drop", "first"),
        sex_c=("sex_c", "first"),
        fat_pct=("fat_pct", "first") if "fat_pct" in d.columns else ("sex_c", "first"),
        date=("date", "first"),
        minute_of_day=("minute_of_day", "first"),
    )

    # If fat_pct didn't exist, the line above duplicated sex_c; fix to NaN
    if "fat_pct" not in d.columns:
        agg["fat_pct"] = np.nan

    agg["dpg"] = agg["skin_temp_ankle"] - agg["skin_temp_neck"]
    return agg


# =========================================================
# QC + Z-OUTLIER FILTER
# =========================================================

def add_site_z_and_filter(df_min: pd.DataFrame, site_col: str, qc_cols) -> pd.DataFrame:
    d = df_min.copy()

    if isinstance(qc_cols, str):
        qc_cols = [qc_cols]

    bad_any = np.zeros(len(d), dtype=bool)
    for qc_col in qc_cols:
        is_bad = normalize_bool_series(d[qc_col], na_is_ok=TREAT_NA_QC_AS_OK)
        bad_any = bad_any | is_bad.to_numpy(dtype=bool)
    d = d.loc[~bad_any].copy()

    d = d.loc[d[site_col].notna()].copy()

    def zscore(g):
        mu = g[site_col].mean()
        sd = g[site_col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            g[f"{site_col}_z_qc"] = 0.0
        else:
            g[f"{site_col}_z_qc"] = (g[site_col] - mu) / sd
        return g

    d = d.groupby(["part_id", "scenario"], group_keys=False).apply(zscore)
    d["is_outlier_site"] = d[f"{site_col}_z_qc"].abs() > float(Z_OUTLIER_THRESH_SITE)
    d = d.loc[~d["is_outlier_site"]].copy()

    return d


# =========================================================
# DRIFT COMPUTATION
# =========================================================

def compute_drift_table_for_site(df_min: pd.DataFrame, site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if site == "neck":
        val_col = "skin_temp_neck"
        qc_cols = ["qc_neck_sudden_drop"]
        label = "Neck"
    elif site == "ankle":
        val_col = "skin_temp_ankle"
        qc_cols = ["qc_ankle_sudden_drop"]
        label = "Ankle"
    elif site == "dpg":
        val_col = "dpg"
        qc_cols = ["qc_neck_sudden_drop", "qc_ankle_sudden_drop"]
        label = "DPG"
    else:
        raise ValueError("site must be 'neck', 'ankle', or 'dpg'")

    d = add_site_z_and_filter(df_min, site_col=val_col, qc_cols=qc_cols)

    d["is_early"] = in_window(d["minute_of_day"], EARLY_START, EARLY_END)
    d["is_late"]  = in_window(d["minute_of_day"],  LATE_START,  LATE_END)

    gb = ["part_id", "arm", "scenario"]

    early = d[d["is_early"]].groupby(gb).agg(
        n_early=(val_col, "size"),
        early_mean=(val_col, "mean"),
    ).reset_index()

    late = d[d["is_late"]].groupby(gb).agg(
        n_late=(val_col, "size"),
        late_mean=(val_col, "mean"),
    ).reset_index()

    drift = early.merge(late, on=gb, how="outer")
    drift["drift"] = drift["late_mean"] - drift["early_mean"]
    drift["site"] = label

    cov = df_min.groupby(["part_id"], as_index=False)[["sex_c", "fat_pct"]].first()
    drift = drift.merge(cov, on="part_id", how="left")

    excl = drift[
        (drift["n_early"].fillna(0) < MIN_VALID_MINUTES_PER_WINDOW) |
        (drift["n_late"].fillna(0)  < MIN_VALID_MINUTES_PER_WINDOW)
    ].copy()
    excl["reason"] = (
        f"{label}_min_minutes_fail("
        + "n_early=" + excl["n_early"].fillna(0).astype(int).astype(str)
        + ",n_late=" + excl["n_late"].fillna(0).astype(int).astype(str)
        + ")"
    )

    drift_ok = drift.drop(excl.index).copy()

    drift_ok["arm"] = pd.Categorical(drift_ok["arm"], categories=ARM_ORDER, ordered=True)
    drift_ok["scenario"] = pd.Categorical(drift_ok["scenario"], categories=SCEN_ORDER, ordered=True)

    return drift_ok, excl


def drop_asymmetrical_participants(drift_df: pd.DataFrame, excl_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if drift_df.empty:
        return drift_df, excl_df, []

    n_scen = drift_df.groupby("part_id")["scenario"].nunique()
    dropped = sorted(n_scen[n_scen < REQUIRED_SCENARIOS_PER_PART].index.astype(str).tolist())
    if not dropped:
        return drift_df, excl_df, []

    add_rows = []
    for pid in dropped:
        have = drift_df.loc[drift_df["part_id"] == pid, "scenario"].astype(str).unique().tolist()
        add_rows.append({
            "part_id": pid,
            "arm": drift_df.loc[drift_df["part_id"] == pid, "arm"].astype(str).iloc[0] if (drift_df["part_id"] == pid).any() else "",
            "scenario": "NA",
            "n_early": np.nan,
            "early_mean": np.nan,
            "n_late": np.nan,
            "late_mean": np.nan,
            "drift": np.nan,
            "site": drift_df["site"].iloc[0] if "site" in drift_df.columns and not drift_df.empty else "",
            "reason": "asymmetrical_missing_scenario(have=" + ",".join(sorted(have)) + ")",
        })

    excl_add = pd.DataFrame(add_rows)
    excl2 = pd.concat([excl_df.copy(), excl_add], ignore_index=True, sort=False)
    kept = drift_df.loc[~drift_df["part_id"].astype(str).isin(dropped)].copy()

    return kept, excl2, dropped


# =========================================================
# MODEL FITTING
# =========================================================

def _fit_mixedlm_with_fallback(formula: str, data: pd.DataFrame, reml: bool) -> tuple[object | None, str | None]:
    """
    Returns (result, error_string). Uses multiple optimizers.
    """
    last_err = None
    for method in ["lbfgs", "powell", "cg", "nm"]:
        try:
            res = smf.mixedlm(formula, data, groups=data["part_id"]).fit(
                reml=reml, method=method, maxiter=2000, disp=False
            )
            return res, None
        except Exception as e:
            last_err = e
            continue
    return None, str(last_err)


def fit_final_model(drift_df: pd.DataFrame):
    """
    Primary (reporting) model, kept aligned with your CBT approach.
    """
    d = drift_df.dropna(subset=["drift", "arm", "scenario", "sex_c", "part_id"]).copy()
    fml = "drift ~ arm * scenario + sex_c"

    res, err = _fit_mixedlm_with_fallback(fml, d, reml=True)
    if res is None:
        raise RuntimeError(f"Final model failed: {err}")
    return res, fml, d


def compute_aic_bic_selection(drift_df: pd.DataFrame, site: str) -> pd.DataFrame:
    """
    Fits 3 models under ML (REML=False) and returns a table with AIC/BIC.
    """
    # Base required
    base = drift_df.dropna(subset=["drift", "arm", "scenario", "part_id"]).copy()

    # Ensure categorical consistency
    base["arm"] = pd.Categorical(base["arm"], categories=ARM_ORDER, ordered=True)
    base["scenario"] = pd.Categorical(base["scenario"], categories=SCEN_ORDER, ordered=True)

    models = [
        ("M0_arm_x_scen", "drift ~ arm * scenario", ["drift", "arm", "scenario"]),
        ("M1_plus_sex",   "drift ~ arm * scenario + sex_c", ["drift", "arm", "scenario", "sex_c"]),
        ("M2_plus_fat",   "drift ~ arm * scenario + sex_c + fat_pct", ["drift", "arm", "scenario", "sex_c", "fat_pct"]),
    ]

    rows = []
    for model_name, formula, req_cols in models:
        d = base.copy()
        for c in req_cols:
            if c not in d.columns:
                d[c] = np.nan
        d = d.dropna(subset=req_cols).copy()

        n_obs = int(d.shape[0])
        n_part = int(d["part_id"].nunique()) if n_obs > 0 else 0

        if n_obs < 6 or n_part < 3:
            rows.append({
                "site": site,
                "model": model_name,
                "formula": formula,
                "reml": False,
                "n_obs": n_obs,
                "n_participants": n_part,
                "aic": np.nan,
                "bic": np.nan,
                "converged": False,
                "error": "insufficient_data_after_na_drop",
            })
            continue

        res, err = _fit_mixedlm_with_fallback(formula, d, reml=False)
        if res is None:
            rows.append({
                "site": site,
                "model": model_name,
                "formula": formula,
                "reml": False,
                "n_obs": n_obs,
                "n_participants": n_part,
                "aic": np.nan,
                "bic": np.nan,
                "converged": False,
                "error": err,
            })
        else:
            rows.append({
                "site": site,
                "model": model_name,
                "formula": formula,
                "reml": False,
                "n_obs": n_obs,
                "n_participants": n_part,
                "aic": float(res.aic) if hasattr(res, "aic") else np.nan,
                "bic": float(res.bic) if hasattr(res, "bic") else np.nan,
                "converged": bool(getattr(res, "converged", True)),
                "error": "",
            })

    out = pd.DataFrame(rows)

    # Add deltas within site (vs M0)
    for metric in ["aic", "bic"]:
        base_val = out.loc[out["model"] == "M0_arm_x_scen", metric].iloc[0]
        out[f"delta_{metric}_vs_M0"] = out[metric] - base_val if np.isfinite(base_val) else np.nan

    # Mark best AIC/BIC within site
    out["best_aic"] = False
    out["best_bic"] = False
    if out["aic"].notna().any():
        out.loc[out["aic"].idxmin(), "best_aic"] = True
    if out["bic"].notna().any():
        out.loc[out["bic"].idxmin(), "best_bic"] = True

    return out


# =========================================================
# MAIN TEXT TABLE
# =========================================================

def build_main_text_table(res, drift_analysis_df: pd.DataFrame) -> pd.DataFrame:
    cov_means = {"sex_c": float(drift_analysis_df["sex_c"].mean())}

    def emm(arm, scen):
        nd = {"arm": arm, "scenario": scen}
        nd.update(cov_means)
        new_df = pd.DataFrame([nd])
        cvec = build_design_row_from_res(res, new_df)
        mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, cvec)
        return mean, se, lo, hi

    def delta_within_arm(arm):
        nd1 = {"arm": arm, "scenario": "HS2"}
        nd0 = {"arm": arm, "scenario": "HS1"}
        nd1.update(cov_means); nd0.update(cov_means)
        c1 = build_design_row_from_res(res, pd.DataFrame([nd1]))
        c0 = build_design_row_from_res(res, pd.DataFrame([nd0]))
        c = c1 - c0
        mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, c)
        return mean, se, tval, p, lo, hi

    wide = drift_analysis_df.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()

    rows = []
    for arm in ARM_ORDER:
        m1, se1, lo1, hi1 = emm(arm, "HS1")
        m2, se2, lo2, hi2 = emm(arm, "HS2")
        dmean, dse, dt, dp, dlo, dhi = delta_within_arm(arm)

        ww = wide.loc[wide["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any").copy()
        if ww.shape[0] >= 2:
            dd = (ww["HS2"] - ww["HS1"]).to_numpy(dtype=float)
            sd = float(np.std(dd, ddof=1))
            dz = float(np.mean(dd) / sd) if sd > 0 else np.nan
            n_dd = int(dd.size)
        else:
            dz = np.nan
            n_dd = 0

        rows.append({
            "arm": arm,
            "n_participants": int(drift_analysis_df.loc[drift_analysis_df["arm"] == arm, "part_id"].nunique()),
            "HS1_adj_mean": m1, "HS1_ci_lo": lo1, "HS1_ci_hi": hi1,
            "HS2_adj_mean": m2, "HS2_ci_lo": lo2, "HS2_ci_hi": hi2,
            "delta_HS2_minus_HS1": dmean, "delta_ci_lo": dlo, "delta_ci_hi": dhi,
            "p_within_arm": dp, "dz_within_arm": dz, "n_dd": n_dd,
        })

    return pd.DataFrame(rows)


# =========================================================
# PLOT
# =========================================================

def plot_paired(drift_df: pd.DataFrame, outdir: str, fname_prefix: str, ylab: str, title: str):
    d = drift_df.dropna(subset=["drift", "arm", "scenario", "part_id"]).copy()
    d["arm"] = pd.Categorical(d["arm"], categories=ARM_ORDER, ordered=True)
    d["scenario"] = pd.Categorical(d["scenario"], categories=SCEN_ORDER, ordered=True)

    x_arm = {arm: i for i, arm in enumerate(ARM_ORDER)}
    x_scen = {"HS1": -0.15, "HS2": 0.15}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    summ = d.groupby(["arm", "scenario"]).agg(n=("drift", "count"), mean=("drift", "mean"), sd=("drift", "std")).reset_index()
    summ["se"] = summ["sd"] / np.sqrt(summ["n"].clip(lower=1))
    summ["ci_lo"] = summ["mean"] - 1.96 * summ["se"]
    summ["ci_hi"] = summ["mean"] + 1.96 * summ["se"]

    rng = np.random.default_rng(0)

    for arm in ARM_ORDER:
        for scen in SCEN_ORDER:
            dd = d[(d["arm"] == arm) & (d["scenario"] == scen)].copy()
            if dd.empty:
                continue
            x0 = x_arm[arm] + x_scen[scen]
            jitter = rng.uniform(-JITTER, JITTER, size=len(dd))
            xs = x0 + jitter

            marker = "o" if scen == "HS1" else "s"
            ax.scatter(xs, dd["drift"].values, s=POINT_SIZE, marker=marker,
                       facecolor="black", edgecolor="black", linewidth=0.6, alpha=0.85, zorder=2)

            if LABEL_PARTICIPANTS:
                for xi, yi, pid in zip(xs, dd["drift"].values, dd["part_id"].astype(str).values):
                    ax.text(xi, yi, pid, fontsize=LABEL_FONTSIZE, ha="left", va="center", color="black")

    if PAIR_LINES:
        wide = d.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()
        for _, r in wide.iterrows():
            arm = r["arm"]
            y1 = r.get("HS1", np.nan)
            y2 = r.get("HS2", np.nan)
            if not (np.isfinite(y1) and np.isfinite(y2)):
                continue
            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, alpha=0.6, zorder=1)

    for _, r in summ.iterrows():
        arm, scen = r["arm"], r["scenario"]
        x0 = x_arm[arm] + x_scen[scen]
        marker = "o" if scen == "HS1" else "s"
        ax.errorbar([x0], [r["mean"]],
                    yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt=marker, color="black", ecolor="black",
                    elinewidth=ERR_LW, capsize=4, markersize=6,
                    markerfacecolor="black", markeredgecolor="black", zorder=4)

    try:
        y_min, y_max = ax.get_ylim()
        y_span = max(1e-9, (y_max - y_min))
        y_pad = 0.06 * y_span
        wide_all = d.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()

        for arm in ARM_ORDER:
            ww = wide_all[wide_all["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any")
            if ww.shape[0] < 3:
                continue
            pre = ww["HS1"].to_numpy(dtype=float)
            post = ww["HS2"].to_numpy(dtype=float)
            diff = post - pre

            _, p_val = stats.ttest_rel(post, pre, nan_policy="omit")
            sd = float(np.std(diff, ddof=1))
            dz = float(np.mean(diff) / sd) if sd > 0 else np.nan

            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            y = max(np.nanmax(pre), np.nanmax(post)) + y_pad

            ax.plot([x1, x1, x2, x2], [y - 0.25*y_pad, y, y, y - 0.25*y_pad],
                    color="black", linewidth=1.0, zorder=5)
            ax.text((x1 + x2) / 2, y + 0.1*y_pad,
                    f"p={p_val:.3f}, dz={dz:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")
    except Exception:
        pass

    ax.axhline(0, color="black", linewidth=AXH0_LW, alpha=0.7)
    ax.set_xticks([x_arm[a] for a in ARM_ORDER])
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=6, label='PRE (HS1)'),
        Line2D([0], [0], marker='s', color='black', linestyle='None', markersize=6, label='POST (HS2)'),
    ]
    ax.legend(handles=handles, frameon=False, loc='best')

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(os.path.join(outdir, f"{fname_prefix}_paired.png"), dpi=300)
    if SAVE_PDF:
        fig.savefig(os.path.join(outdir, f"{fname_prefix}_paired.pdf"), format="pdf")

    plt.close(fig)


# =========================================================
# RUN
# =========================================================

def run_site(site: str, df_min: pd.DataFrame):
    drift_df, excl_df = compute_drift_table_for_site(df_min, site=site)

    if DROP_ASYMMETRICAL_PARTICIPANTS:
        drift_df, excl_df, dropped_asym = drop_asymmetrical_participants(drift_df, excl_df)
        if dropped_asym:
            print(f"Dropped asymmetrical participants for {site} (missing HS1/HS2):", dropped_asym)

    drift_df.to_csv(os.path.join(OUTDIR, f"SkinTemp_{site}_drift_participant_level.csv"), index=False)
    excl_df.to_csv(os.path.join(OUTDIR, f"SkinTemp_{site}_drift_exclusions.csv"), index=False)

    # Primary model (REML) + outputs
    res, fml, drift_analysis_df = fit_final_model(drift_df)

    with open(os.path.join(OUTDIR, f"SkinTemp_{site}_final_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Final model formula:\n")
        f.write(fml + "\n\n")
        f.write(str(res.summary()))

    main_tbl = build_main_text_table(res, drift_analysis_df)
    main_tbl.to_csv(os.path.join(OUTDIR, f"SkinTemp_{site}_Table_MainText.csv"), index=False)
    main_tbl.to_string(open(os.path.join(OUTDIR, f"SkinTemp_{site}_Table_MainText.txt"), "w", encoding="utf-8"), index=False)

    # Plot labels
    if site == "neck":
        ylab = "Neck skin temperature drift (Late − Early), °C"
        title = "Neck skin temperature drift by arm and scenario (HS1 vs HS2)"
    elif site == "ankle":
        ylab = "Ankle skin temperature drift (Late − Early), °C"
        title = "Ankle skin temperature drift by arm and scenario (HS1 vs HS2)"
    else:
        ylab = "DPG drift (Late − Early), °C"
        title = "DPG drift by arm and scenario (HS1 vs HS2)"

    plot_paired(drift_analysis_df, OUTDIR, fname_prefix=f"SkinTemp_{site}", ylab=ylab, title=title)

    return drift_df, excl_df, res


def run_analysis():
    df, part, filt_log = load_and_prepare()

    filt_log.to_csv(os.path.join(OUTDIR, "SkinTemp_filtering_log.csv"), index=False)

    df_min = resample_to_minutes(df)

    print("\n=== MINUTE-LEVEL DATASET (after exclusions + clipping) ===")
    print("Rows:", df_min.shape[0])
    print("Participants:", df_min["part_id"].nunique())
    print("Arms:", df_min["arm"].value_counts(dropna=False).to_dict())
    print("Scenarios:", df_min["scenario"].value_counts(dropna=False).to_dict())

    model_sel_rows = []

    for site in ["neck", "ankle", "dpg"]:
        print(f"\n=== RUN SITE: {site.upper()} ===")
        drift_df, excl_df, res = run_site(site, df_min)

        print("Rows (part×scenario):", drift_df.shape[0])
        print("Participants:", drift_df["part_id"].nunique())
        print("Counts arm×scenario:\n", pd.crosstab(drift_df["arm"], drift_df["scenario"]))

        # NEW: AIC/BIC selection fits (ML) for this site
        sel = compute_aic_bic_selection(drift_df, site=site)
        model_sel_rows.append(sel)

    # NEW: write combined model selection CSV across the three signals
    if model_sel_rows:
        model_sel = pd.concat(model_sel_rows, ignore_index=True)
        out_path = os.path.join(OUTDIR, "SkinTemp_model_selection_AIC_BIC.csv")
        model_sel.to_csv(out_path, index=False)
        print("\n[MODEL SELECTION] Wrote:", out_path)

    print("\n[DONE] Outputs written to:", OUTDIR)


def main():
    warnings.filterwarnings("ignore")
    run_analysis()


if __name__ == "__main__":
    main()
