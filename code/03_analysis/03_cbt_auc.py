# -*- coding: utf-8 -*-
"""Heat Acclimation — CBT AUC end-to-end pipeline (cleanup + modelling + exports).

This script is the AUC analogue of your CBT drift pipeline. It reuses the same:
- exclusions (P05 global; P02 on 2023-06-05),
- session start overrides (date-level + one date×participant override),
- session clipping,
- 1-minute resampling,
- QC-flag filtering,
- within (participant×scenario) z-score outlier removal (for QC only),
- optional post-window completeness QC (drop participants missing HS1 or HS2).

Primary estimand (AUC):
-----------------------
Baseline:
  - baseline_mean = mean CBT during EARLY_START–EARLY_END (default 12:15–12:30)

AUC window:
  - default AUC_START–AUC_END = 12:30–16:30

Metrics:
  - auc_delta_h (°C·h)    = ∫(CBT - baseline_mean) dt over AUC window
  - auc_mean_delta (°C)   = auc_delta_h / duration_hours (time-adjusted mean elevation)

Inference:
----------
Mixed effects model on TARGET_METRIC:
    y ~ arm * scenario + covariates + (1|part_id)

Exports:
--------
- CBT_filtering_log.csv
- CBT_AUC_participant_level.csv
- CBT_AUC_exclusions.csv
- CBT_AUC_model_selection_AIC_BIC.csv
- CBT_AUC_final_model_summary.txt
- CBT_AUC_Table_MainText.csv (+ .txt)
- CBT_<metric>_paired.png/.pdf (paired plot with model-consistent means/CIs + model-based p-values)

Important consistency note:
---------------------------
The paired plot annotations use model-based within-arm contrasts (HS2–HS1) when res is provided,
so p-values in the figure align with the model outputs (unlike a raw paired t-test).

"""

from __future__ import annotations

from pathlib import Path
import os
import re
import warnings
from pathlib import Path

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
OUTPUTS = REPO_ROOT / "outputs" / "model_outputs" / "cbt_outputs" / "cbt_auc"

# =========================================================
# CONFIG
# =========================================================

# ---- Analysis paths ----
PATH_CBT = DATA_PROCESSED / "cbt.csv"
PATH_PART = DATA_METADATA / "participant_meta_pseud.csv"

OUTDIR = OUTPUTS / "cbt" / "auc_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- required columns ----
REQ_CBT = ["datetime", "part_id", "condition", "scenario_short", "session_id", "cbt_raw", "cbt_flag"]

# Participant meta: required
REQ_PART_BASE = ["part_id", "sex"]

# ---- Arms / scenarios ----
VALID_ARMS = ["FR", "CC"]
ARM_ORDER = ["FR", "CC"]  # plotting/reference

VALID_SCENARIO_SHORT = ["HS_pre", "HS_post"]
SCENARIO_MAP = {"HS_pre": "HS1", "HS_post": "HS2"}
SCEN_ORDER = ["HS1", "HS2"]

# ---- Windows ----
# Baseline window (same as drift)
EARLY_START = "12:15"
EARLY_END = "12:30"

# AUC window (default: immediately after baseline to end)
AUC_START = "12:30"
AUC_END = "16:30"

# ---- Valid minutes thresholds ----
MIN_VALID_MINUTES_BASELINE = 10
MIN_VALID_MINUTES_AUC = 120  # stricter, because AUC needs stable coverage; adjust if needed

# ---- Metric choice ----
# "auc_delta_h" (°C·h) or "auc_mean_delta" (°C)
TARGET_METRIC = "auc_delta_h"

# ---- Resampling ----
RESAMPLE_TO_MINUTE = True

# ---- QC ----
Z_OUTLIER_THRESH = 4.0

# ---- Post-window completeness QC ----
DROP_ASYMMETRICAL_PARTICIPANTS = True
REQUIRED_SCENARIOS_PER_PART = 2  # HS1 and HS2

# ---- Plot ----
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
# PRE-MODEL FILTER RULES (same as drift script)
# =========================================================

# 1) Exclude participant P05 everywhere (dropped out)
EXCLUDE_PARTICIPANTS_ALL = {"P05"}

# 2) Exclude P02 only on 2023-06-05
EXCLUDE_PARTICIPANTS_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): {"P02"},
}

# 3) Session start overrides (date-level)
SESSION_START_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): "10:40",
    pd.Timestamp("2023-06-06").normalize(): "09:55",
    pd.Timestamp("2023-06-07").normalize(): "09:40",
}

# 4) Participant-specific start overrides (date + part)
SESSION_START_BY_DATE_PART = {
    (pd.Timestamp("2023-06-22").normalize(), "P06"): "10:30",
}

# default session end
DEFAULT_SESSION_END = "16:30"


# =========================================================
# GENERIC HELPERS
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
    """Apply participant exclusions and clip each (date[,participant]) to start->DEFAULT_SESSION_END.

    Returns
    -------
    df_kept : DataFrame
    log    : DataFrame of exclusions/clipping summary
    """
    df = df.copy()
    df = add_time_fields(df)

    log_rows = []

    # Exclude participants globally
    if EXCLUDE_PARTICIPANTS_ALL:
        m = df["part_id"].astype(str).isin(list(EXCLUDE_PARTICIPANTS_ALL))
        if m.any():
            for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
                log_rows.append({"action": "exclude_all", "part_id": pid, "date": pd.NaT,
                                 "reason": "dropped_out", "n_rows_clipped": np.nan})
        df = df.loc[~m].copy()

    # Exclude participants by date
    for d, pids in EXCLUDE_PARTICIPANTS_BY_DATE.items():
        m = (df["date"] == pd.Timestamp(d).normalize()) & (df["part_id"].astype(str).isin(list(pids)))
        if m.any():
            for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
                log_rows.append({"action": "exclude_date", "part_id": pid, "date": pd.Timestamp(d).normalize(),
                                 "reason": "manual_exclusion", "n_rows_clipped": np.nan})
        df = df.loc[~m].copy()

    # Clip by (date, part_id) start overrides
    def row_keep(r):
        d = r["date"]
        pid = str(r["part_id"])
        start_hhmm = _get_start_hhmm_for_row(d, pid)
        start_m = clock_to_minutes(start_hhmm)
        end_m = clock_to_minutes(DEFAULT_SESSION_END)
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

    log = pd.DataFrame(log_rows, columns=["action", "part_id", "date", "reason", "n_rows_clipped"])
    return df, log


# =========================================================
# QC: WITHIN-SESSION Z-SCORE (OUTLIER DETECTION ONLY)
# =========================================================

def add_within_session_zscore_qc(df: pd.DataFrame, z_thresh=4.0, value_col="cbt_raw") -> pd.DataFrame:
    """Compute within (participant×scenario) z-score for outlier detection only."""
    df = df.copy()

    def zscore(g):
        mu = g[value_col].mean()
        sd = g[value_col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            g["cbt_z_qc"] = 0.0
        else:
            g["cbt_z_qc"] = (g[value_col] - mu) / sd
        return g

    df = df.groupby(["part_id", "scenario"], group_keys=False).apply(zscore)
    df["is_outlier"] = df["cbt_z_qc"].abs() > float(z_thresh)
    return df


# =========================================================
# MIXEDLM CONTRAST UTILITIES (FIXED EFFECTS)
# =========================================================

def build_design_row_from_res(res, new_df: pd.DataFrame) -> np.ndarray:
    X = dmatrix(res.model.data.design_info, new_df, return_type="dataframe")
    return X.iloc[0].values


def estimate_fe_linear_combo(res, cvec: np.ndarray):
    """Estimate c' beta, SE, t, p, CI using fixed effects and res.cov_params()."""
    beta = np.asarray(res.fe_params).reshape(-1, 1)
    k = beta.shape[0]

    Vfull = np.asarray(res.cov_params())
    V = Vfull[:k, :k]

    c = np.asarray(cvec).reshape(-1, 1)
    if c.shape[0] != k:
        raise ValueError(f"Contrast length {c.shape[0]} does not match #fixed effects {k}")

    mean = float((c.T @ beta)[0, 0])
    var = float((c.T @ V @ c)[0, 0])
    se = float(np.sqrt(max(var, 0.0)))

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
# LOAD + PREP
# =========================================================

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- CBT ---
    cbt = pd.read_csv(PATH_CBT)
    cbt.columns = [c.strip() for c in cbt.columns]
    require_columns(cbt, REQ_CBT, "CBT")

    cbt["part_id"] = cbt["part_id"].astype(str)
    cbt["scenario_short"] = cbt["scenario_short"].astype(str)
    cbt["condition"] = cbt["condition"].astype(str)
    cbt = ensure_numeric(cbt, ["cbt_raw"])

    # keep valid arms/scenarios
    cbt = cbt[cbt["condition"].isin(VALID_ARMS)].copy()
    cbt = cbt[cbt["scenario_short"].isin(VALID_SCENARIO_SHORT)].copy()
    cbt["scenario"] = cbt["scenario_short"].map(SCENARIO_MAP)

    # apply exclusions + clipping BEFORE windowing/resampling
    cbt, filt_log = apply_exclusions_and_session_clipping(cbt)

    # # apply QC flag filter (same logic as drift script)
    # if "cbt_flag" in cbt.columns:
    #     cbt = cbt.copy()
    #     flag = cbt["cbt_flag"]
    #     if flag.dtype == object:
    #         flag_norm = flag.astype(str).str.strip().str.lower()
    #         ok_mask = flag_norm.eq("ok") | flag_norm.eq("0") | flag_norm.eq("nan")
    #         if flag.isna().all():
    #             ok_mask = pd.Series([True] * len(cbt), index=cbt.index)
    #         else:
    #             ok_mask = ok_mask | flag.isna()
    #         cbt = cbt.loc[ok_mask].copy()
    #     else:
    #         flag_num = pd.to_numeric(flag, errors="coerce")
    #         if not flag_num.isna().all():
    #             cbt = cbt.loc[flag_num == 0].copy()

    # --- participant meta ---
    part = pd.read_csv(PATH_PART)
    part.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in part.columns]
    require_columns(part, REQ_PART_BASE, "Participants")

    part["part_id"] = part["part_id"].astype(str)

    # keep only needed columns if present
    keep_cols = ["part_id"] + [c for c in ["sex", "fat_pct", "bmr", "mens_change"] if c in part.columns]
    part = part[keep_cols].copy()

    # recode sex
    part["sex"] = part["sex"].astype(str).str.strip().str.upper()
    part["sex_c"] = part["sex"].map({"F": 1, "M": 0, "FEMALE": 1, "MALE": 0}).astype(float)

    # numeric covariates
    for c in ["fat_pct", "bmr", "mens_change"]:
        if c in part.columns:
            part[c] = pd.to_numeric(part[c], errors="coerce")

    # merge
    df = cbt.merge(part, on="part_id", how="left")

    # centered covariates
    if "fat_pct" in df.columns:
        df["fat_pct_c"] = df["fat_pct"] - df["fat_pct"].mean(skipna=True)
    if "bmr" in df.columns:
        df["bmr_c"] = df["bmr"] - df["bmr"].mean(skipna=True)

    # menstrual term: apply only to females via interaction; set missing/males to 0 to avoid NA drop
    if "mens_change" in df.columns:
        mc = df["mens_change"].copy().fillna(0.0)
        df["mens_change_c"] = mc - mc.mean(skipna=True)
        df["mens_change_f"] = df["mens_change_c"] * df["sex_c"].fillna(0.0)

    return df, part, filt_log


def resample_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if not RESAMPLE_TO_MINUTE:
        # ensure time fields exist
        d0 = df.copy()
        if "datetime" not in d0.columns and "minute_floor" in d0.columns:
            d0["datetime"] = d0["minute_floor"]
        return add_time_fields(d0)

    d = df.copy()
    d = add_time_fields(d)

    # group by minute
    gcols = ["part_id", "condition", "scenario", "session_id", "minute_floor"]
    agg = d.groupby(gcols, as_index=False).agg(
        cbt_raw=("cbt_raw", "mean"),
        sex_c=("sex_c", "first"),
        fat_pct_c=("fat_pct_c", "first") if "fat_pct_c" in d.columns else ("sex_c", "first"),
        bmr_c=("bmr_c", "first") if "bmr_c" in d.columns else ("sex_c", "first"),
        mens_change_f=("mens_change_f", "first") if "mens_change_f" in d.columns else ("sex_c", "first"),
        date=("date", "first"),
        minute_of_day=("minute_of_day", "first"),
    )

    # clean dummy fillers if covariates missing
    if "fat_pct_c" not in d.columns:
        agg = agg.drop(columns=["fat_pct_c"], errors="ignore")
    if "bmr_c" not in d.columns:
        agg = agg.drop(columns=["bmr_c"], errors="ignore")
    if "mens_change_f" not in d.columns:
        agg = agg.drop(columns=["mens_change_f"], errors="ignore")

    # IMPORTANT: reintroduce datetime so downstream helpers work unchanged
    agg["datetime"] = agg["minute_floor"]

    return agg



# =========================================================
# AUC COMPUTATION
# =========================================================

def compute_auc_table(df_min: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return AUC metrics per participant×arm×scenario and an exclusion table."""
    d = df_min.copy()
    d = d.rename(columns={"condition": "arm"})
    d = add_time_fields(d)

    # outlier QC (within participant×scenario)
    tmp = d.rename(columns={"arm": "condition"})
    tmp = add_within_session_zscore_qc(tmp, z_thresh=Z_OUTLIER_THRESH, value_col="cbt_raw")
    d = tmp.rename(columns={"condition": "arm"})
    d = d[~d["is_outlier"]].copy()

    d["is_base"] = in_window(d["minute_of_day"], EARLY_START, EARLY_END)
    d["is_auc"] = in_window(d["minute_of_day"], AUC_START, AUC_END)

    gb = ["part_id", "arm", "scenario"]

    base = d[d["is_base"]].groupby(gb).agg(
        n_base=("cbt_raw", "size"),
        base_mean=("cbt_raw", "mean"),
    ).reset_index()

    # compute pointwise deltas in the AUC window
    auc_points = d[d["is_auc"]].merge(base, on=gb, how="left")
    auc_points["delta"] = auc_points["cbt_raw"] - auc_points["base_mean"]

    auc = auc_points.groupby(gb).agg(
        n_auc=("delta", "size"),
        auc_delta_min=("delta", "sum"),  # °C·min since dt=1 min bins
    ).reset_index()

    out = base.merge(auc, on=gb, how="outer")

    # convert units
    out["auc_delta_h"] = out["auc_delta_min"] / 60.0  # °C·h
    auc_dur_h = (clock_to_minutes(AUC_END) - clock_to_minutes(AUC_START)) / 60.0
    out["auc_mean_delta"] = out["auc_delta_h"] / auc_dur_h

    # attach covariates (first available from df_min)
    cov_cols = [c for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"] if c in df_min.columns]
    cov = df_min.groupby(["part_id"], as_index=False)[cov_cols].first() if cov_cols else df_min[["part_id"]].drop_duplicates()
    out = out.merge(cov, on="part_id", how="left")

    # exclusions due to insufficient minutes
    excl = out[
        (out["n_base"].fillna(0) < MIN_VALID_MINUTES_BASELINE) |
        (out["n_auc"].fillna(0) < MIN_VALID_MINUTES_AUC)
    ].copy()

    excl["reason"] = (
        "min_minutes_fail(" +
        "n_base=" + excl["n_base"].fillna(0).astype(int).astype(str) +
        ",n_auc=" + excl["n_auc"].fillna(0).astype(int).astype(str) +
        ")"
    )

    out_ok = out.drop(excl.index).copy()

    # ordering
    out_ok["arm"] = pd.Categorical(out_ok["arm"], categories=ARM_ORDER, ordered=True)
    out_ok["scenario"] = pd.Categorical(out_ok["scenario"], categories=SCEN_ORDER, ordered=True)

    return out_ok, excl


# =========================================================
# POST-WINDOW QC: DROP ASYMMETRICAL PARTICIPANTS
# =========================================================

def drop_asymmetrical_participants(metric_df: pd.DataFrame, excl_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop participants who are missing HS1 or HS2 after AUC computation."""
    if metric_df.empty:
        return metric_df, excl_df, []

    n_scen = metric_df.groupby("part_id")["scenario"].nunique()
    dropped = sorted(n_scen[n_scen < REQUIRED_SCENARIOS_PER_PART].index.astype(str).tolist())
    if not dropped:
        return metric_df, excl_df, []

    add_rows = []
    for pid in dropped:
        have = metric_df.loc[metric_df["part_id"] == pid, "scenario"].astype(str).unique().tolist()
        add_rows.append({
            "part_id": pid,
            "arm": metric_df.loc[metric_df["part_id"] == pid, "arm"].astype(str).iloc[0] if (metric_df["part_id"] == pid).any() else "",
            "scenario": "NA",
            "n_base": np.nan,
            "base_mean": np.nan,
            "n_auc": np.nan,
            "auc_delta_min": np.nan,
            "auc_delta_h": np.nan,
            "auc_mean_delta": np.nan,
            "reason": "asymmetrical_missing_scenario(have=" + ",".join(sorted(have)) + ")",
        })

    excl_add = pd.DataFrame(add_rows)
    excl_df2 = pd.concat([excl_df.copy(), excl_add], ignore_index=True, sort=False)

    kept = metric_df.loc[~metric_df["part_id"].astype(str).isin(dropped)].copy()
    return kept, excl_df2, dropped


# =========================================================
# MODEL SELECTION
# =========================================================

def model_selection(metric_df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """
    Compare covariate sets via AIC/BIC for the AUC model.

    Notes:
    - Fits MixedLM with participant random intercept (groups=part_id).
    - Tries multiple optimizers; if all fail, records the last error for that model.
    - Drops rows with missing variables required by each formula.
    """
    base_terms = "arm * scenario"

    candidates = []
    if "fat_pct_c" in metric_df.columns and metric_df["fat_pct_c"].notna().sum() >= 10:
        candidates.append("fat_pct_c")
    if "bmr_c" in metric_df.columns and metric_df["bmr_c"].notna().sum() >= 10:
        candidates.append("bmr_c")
    if "mens_change_f" in metric_df.columns and metric_df["mens_change_f"].notna().sum() >= 10:
        candidates.append("mens_change_f")

    always = []
    if "sex_c" in metric_df.columns and metric_df["sex_c"].notna().sum() >= 10:
        always.append("sex_c")

    def _terms_str(terms):
        return (" + " + " + ".join(terms)) if terms else ""

    def _fit_mixedlm_multiopt(fml: str, d: pd.DataFrame):
        last_err = None
        for method in ["lbfgs", "powell", "cg", "nm"]:
            try:
                res = smf.mixedlm(fml, d, groups=d["part_id"]).fit(
                    reml=False, method=method, maxiter=2000, disp=False
                )
                return res
            except Exception as e:
                last_err = e
                continue
        raise last_err

    formulas = []
    formulas.append(("M0_base", f"{ycol} ~ {base_terms}{_terms_str(always)}"))

    if "fat_pct_c" in candidates and "bmr_c" in candidates:
        formulas.append(("M1_plus_fat", f"{ycol} ~ {base_terms}{_terms_str(always + ['fat_pct_c'])}"))
        formulas.append(("M2_plus_bmr", f"{ycol} ~ {base_terms}{_terms_str(always + ['bmr_c'])}"))
        formulas.append(("M3_plus_fat_bmr", f"{ycol} ~ {base_terms}{_terms_str(always + ['fat_pct_c', 'bmr_c'])}"))
    elif "fat_pct_c" in candidates:
        formulas.append(("M1_plus_fat", f"{ycol} ~ {base_terms}{_terms_str(always + ['fat_pct_c'])}"))
    elif "bmr_c" in candidates:
        formulas.append(("M1_plus_bmr", f"{ycol} ~ {base_terms}{_terms_str(always + ['bmr_c'])}"))

    if "mens_change_f" in candidates:
        extra_covs = [c for c in ["fat_pct_c", "bmr_c"] if c in candidates]
        formulas.append(("M4_plus_mens", f"{ycol} ~ {base_terms}{_terms_str(always + extra_covs + ['mens_change_f'])}"))

    rows = []
    for name, fml in formulas:
        d = metric_df.copy()

        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", fml)
        ignore = {ycol, "arm", "scenario"}
        vars_needed = [t for t in tokens if t not in ignore and t in d.columns]

        use_cols = list(dict.fromkeys([ycol, "part_id", "arm", "scenario"] + vars_needed))
        d = d[use_cols].dropna().copy()

        n_part = int(d["part_id"].nunique()) if not d.empty else 0
        if n_part < 3:
            rows.append({
                "model": name,
                "formula": fml,
                "n": n_part,
                "aic": np.nan,
                "bic": np.nan,
                "error": f"Insufficient participants after dropna (n={n_part})",
            })
            continue

        try:
            res = _fit_mixedlm_multiopt(fml, d)
            rows.append({
                "model": name,
                "formula": fml,
                "n": n_part,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "error": "",
            })
        except Exception as e:
            rows.append({
                "model": name,
                "formula": fml,
                "n": n_part,
                "aic": np.nan,
                "bic": np.nan,
                "error": str(e),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        if out["aic"].notna().any():
            out["delta_aic_vs_best"] = out["aic"] - out["aic"].min(skipna=True)
        else:
            out["delta_aic_vs_best"] = np.nan
        if out["bic"].notna().any():
            out["delta_bic_vs_best"] = out["bic"] - out["bic"].min(skipna=True)
        else:
            out["delta_bic_vs_best"] = np.nan

    return out.sort_values(["delta_aic_vs_best", "delta_bic_vs_best"], na_position="last")


def fit_final_model(metric_df: pd.DataFrame, model_table: pd.DataFrame, ycol: str):
    """Pick best (lowest AIC) successful model; fall back to base."""
    pick = model_table.dropna(subset=["aic"]).sort_values("aic").head(1)
    if pick.empty:
        fml = f"{ycol} ~ arm * scenario"
    else:
        fml = str(pick.iloc[0]["formula"])

    vars_needed = [v for v in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", fml) if v not in {ycol}]
    use_cols = list(set([ycol, "part_id"] + vars_needed))
    d = metric_df[use_cols].dropna().copy()

    last_err = None
    for method in ["lbfgs", "powell", "cg", "nm"]:
        try:
            res = smf.mixedlm(fml, d, groups=d["part_id"]).fit(reml=True, method=method, maxiter=2000, disp=False)
            return res, fml, d
        except Exception as e:
            last_err = e
            continue
    raise last_err


# =========================================================
# MAIN TEXT TABLE
# =========================================================

def build_main_text_table(res, analysis_df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """Create main-text table: HS1/HS2 adjusted means + within-arm Δ + dz."""
    # compute EMMs for each arm × scenario at mean covariates
    cov_means = {}
    for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"]:
        if c in analysis_df.columns:
            cov_means[c] = float(analysis_df[c].mean())

    # helper: adjusted mean for a given arm/scenario
    def emm(arm, scen):
        nd = {"arm": arm, "scenario": scen}
        nd.update(cov_means)
        cvec = build_design_row_from_res(res, pd.DataFrame([nd]))
        mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, cvec)
        return mean, lo, hi

    # within-arm Δ: HS2 - HS1
    def delta_within_arm(arm):
        nd1 = {"arm": arm, "scenario": "HS2"}
        nd0 = {"arm": arm, "scenario": "HS1"}
        nd1.update(cov_means)
        nd0.update(cov_means)
        c1 = build_design_row_from_res(res, pd.DataFrame([nd1]))
        c0 = build_design_row_from_res(res, pd.DataFrame([nd0]))
        c = c1 - c0
        mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, c)
        return mean, p, lo, hi

    # dz from participant DD distribution (raw)
    wide = analysis_df.pivot_table(index=["part_id", "arm"], columns="scenario", values=ycol, aggfunc="first").reset_index()

    rows = []
    for arm in ARM_ORDER:
        m1, lo1, hi1 = emm(arm, "HS1")
        m2, lo2, hi2 = emm(arm, "HS2")
        dmean, dp, dlo, dhi = delta_within_arm(arm)

        ww = wide.loc[wide["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any")
        if ww.shape[0] >= 2:
            dd = (ww["HS2"] - ww["HS1"]).to_numpy(dtype=float)
            sd = float(np.std(dd, ddof=1))
            dz = float(np.mean(dd) / sd) if sd > 0 else np.nan
        else:
            dz = np.nan

        rows.append({
            "arm": arm,
            "n_participants": int(analysis_df.loc[analysis_df["arm"] == arm, "part_id"].nunique()),
            "HS1_adj_mean": m1,
            "HS2_adj_mean": m2,
            "delta_HS2_minus_HS1": dmean,
            "delta_ci_lo": dlo,
            "delta_ci_hi": dhi,
            "p_within_arm": dp,
            "dz_within_arm": dz,
            "HS1_ci_lo": lo1,
            "HS1_ci_hi": hi1,
            "HS2_ci_lo": lo2,
            "HS2_ci_hi": hi2,
        })

    return pd.DataFrame(rows)


# =========================================================
# MODEL-CONSISTENT PAIRED PLOT
# =========================================================

def plot_paired(metric_df: pd.DataFrame, outdir: str, ycol: str, res=None, cov_source_df: pd.DataFrame | None = None):
    """Paired points plot HS1/HS2 per arm with model-consistent summary bars and p-values."""
    d = metric_df.copy()
    d = d.dropna(subset=[ycol, "arm", "scenario", "part_id"]).copy()

    # enforce ordering
    d["arm"] = pd.Categorical(d["arm"], categories=ARM_ORDER, ordered=True)
    d["scenario"] = pd.Categorical(d["scenario"], categories=SCEN_ORDER, ordered=True)

    # numeric x positions
    x_arm = {arm: i for i, arm in enumerate(ARM_ORDER)}
    x_scen = {"HS1": -0.15, "HS2": 0.15}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    # participant points
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
            ax.scatter(
                xs,
                dd[ycol].values,
                s=POINT_SIZE,
                marker=marker,
                facecolor="black",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
                zorder=2,
            )

            if LABEL_PARTICIPANTS:
                for xi, yi, pid in zip(xs, dd[ycol].values, dd["part_id"].astype(str).values):
                    ax.text(xi, yi, pid, fontsize=LABEL_FONTSIZE, ha="left", va="center", color="black")

    # pair lines
    if PAIR_LINES:
        wide = d.pivot_table(index=["part_id", "arm"], columns="scenario", values=ycol, aggfunc="first").reset_index()
        for _, r in wide.iterrows():
            arm = r["arm"]
            y1 = r.get("HS1", np.nan)
            y2 = r.get("HS2", np.nan)
            if not (np.isfinite(y1) and np.isfinite(y2)):
                continue
            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, alpha=0.6, zorder=1)

    # summary means + CIs: model-based if res provided, else raw mean CI
    use_model = res is not None
    if use_model:
        if cov_source_df is None:
            cov_source_df = d
        cov_means = {}
        for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"]:
            if c in cov_source_df.columns:
                cov_means[c] = float(cov_source_df[c].mean())

        def emm_mean_ci(arm, scen):
            nd = {"arm": arm, "scenario": scen}
            nd.update(cov_means)
            cvec = build_design_row_from_res(res, pd.DataFrame([nd]))
            mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, cvec)
            return mean, lo, hi

        summ_rows = []
        for arm in ARM_ORDER:
            for scen in SCEN_ORDER:
                m, lo, hi = emm_mean_ci(arm, scen)
                summ_rows.append({"arm": arm, "scenario": scen, "mean": m, "ci_lo": lo, "ci_hi": hi})
        summ = pd.DataFrame(summ_rows)
    else:
        summ = d.groupby(["arm", "scenario"]).agg(
            n=(ycol, "count"),
            mean=(ycol, "mean"),
            sd=(ycol, "std"),
        ).reset_index()
        summ["se"] = summ["sd"] / np.sqrt(summ["n"].clip(lower=1))
        summ["ci_lo"] = summ["mean"] - 1.96 * summ["se"]
        summ["ci_hi"] = summ["mean"] + 1.96 * summ["se"]

    for _, r in summ.iterrows():
        arm, scen = r["arm"], r["scenario"]
        x0 = x_arm[arm] + x_scen[scen]
        marker = "o" if scen == "HS1" else "s"
        ax.errorbar(
            [x0],
            [r["mean"]],
            yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
            fmt=marker,
            color="black",
            ecolor="black",
            elinewidth=ERR_LW,
            capsize=4,
            markersize=6,
            markerfacecolor="black",
            markeredgecolor="black",
            zorder=4,
        )

    # bracket annotations: model-based within-arm p + dz from DD distribution
    try:
        y_min, y_max = ax.get_ylim()
        y_span = max(1e-9, (y_max - y_min))
        y_pad = 0.06 * y_span

        wide_all = d.pivot_table(index=["part_id", "arm"], columns="scenario", values=ycol, aggfunc="first").reset_index()

        def dz_from_dd(arm):
            ww = wide_all[wide_all["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any")
            if ww.shape[0] < 2:
                return np.nan
            diff = (ww["HS2"] - ww["HS1"]).to_numpy(dtype=float)
            sd = float(np.std(diff, ddof=1))
            return float(np.mean(diff) / sd) if sd > 0 else np.nan

        def p_within_arm_model(arm):
            if not use_model:
                return np.nan
            if cov_source_df is None:
                cov_source_df = d
            cov_means = {}
            for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"]:
                if c in cov_source_df.columns:
                    cov_means[c] = float(cov_source_df[c].mean())

            nd1 = {"arm": arm, "scenario": "HS2"}; nd1.update(cov_means)
            nd0 = {"arm": arm, "scenario": "HS1"}; nd0.update(cov_means)
            c1 = build_design_row_from_res(res, pd.DataFrame([nd1]))
            c0 = build_design_row_from_res(res, pd.DataFrame([nd0]))
            mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, c1 - c0)
            return float(p)

        for arm in ARM_ORDER:
            ww = wide_all[wide_all["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any")
            if ww.shape[0] < 3:
                continue

            pre = ww["HS1"].to_numpy(dtype=float)
            post = ww["HS2"].to_numpy(dtype=float)

            y = max(np.nanmax(pre), np.nanmax(post)) + y_pad
            dz = dz_from_dd(arm)

            if use_model:
                p_val = p_within_arm_model(arm)
            else:
                _, p_val = stats.ttest_rel(post, pre, nan_policy="omit")
                p_val = float(p_val)

            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            ax.plot([x1, x1, x2, x2], [y - 0.25 * y_pad, y, y, y - 0.25 * y_pad], color="black", linewidth=1.0, zorder=5)
            ax.text((x1 + x2) / 2, y + 0.1 * y_pad, f"p={p_val:.3f}, dz={dz:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")

    except Exception:
        pass

    ax.axhline(0, color="black", linewidth=AXH0_LW, alpha=0.7)
    ax.set_xticks([x_arm[a] for a in ARM_ORDER])
    ax.set_xticklabels(ARM_ORDER)

    if ycol == "auc_delta_h":
        ax.set_ylabel("CBT AUCΔ (relative to baseline), °C·h")
    else:
        ax.set_ylabel("CBT mean Δ (relative to baseline), °C")

    ax.set_title("CBT AUC by arm and scenario (HS1 vs HS2)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=6, label='PRE (HS1)'),
        Line2D([0], [0], marker='s', color='black', linestyle='None', markersize=6, label='POST (HS2)'),
    ]
    ax.legend(handles=handles, frameon=False, loc='best')

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(os.path.join(outdir, f"CBT_{ycol}_paired.png"), dpi=300)
    if SAVE_PDF:
        fig.savefig(os.path.join(outdir, f"CBT_{ycol}_paired.pdf"), format="pdf")

    plt.close(fig)


# =========================================================
# RUNNER
# =========================================================

def run_analysis():
    df, part, filt_log = load_and_prepare()

    # export filter log
    filt_log.to_csv(os.path.join(OUTDIR, "CBT_filtering_log.csv"), index=False)

    # minute resample
    df_min = resample_to_minutes(df)

    # compute AUC metrics
    metric_df, excl_df = compute_auc_table(df_min)

    # Drop participants with missing HS1 or HS2 after AUC computation (optional)
    if DROP_ASYMMETRICAL_PARTICIPANTS:
        metric_df, excl_df, dropped_asym = drop_asymmetrical_participants(metric_df, excl_df)
        if dropped_asym:
            print("Dropped asymmetrical participants (missing HS1 or HS2):", dropped_asym)

    # choose metric column
    if TARGET_METRIC not in {"auc_delta_h", "auc_mean_delta"}:
        raise ValueError("TARGET_METRIC must be 'auc_delta_h' or 'auc_mean_delta'")
    ycol = TARGET_METRIC

    print("\n=== AUC TABLE (after filtering + window requirements) ===")
    print("Rows (part×scenario):", metric_df.shape[0])
    print("Participants:", metric_df["part_id"].nunique())
    print("Counts arm×scenario:\n", pd.crosstab(metric_df["arm"], metric_df["scenario"]))

    # exports: participant-level + exclusions
    metric_df.to_csv(os.path.join(OUTDIR, "CBT_AUC_participant_level.csv"), index=False)
    excl_df.to_csv(os.path.join(OUTDIR, "CBT_AUC_exclusions.csv"), index=False)

    # model selection
    ms = model_selection(metric_df, ycol=ycol)
    ms.to_csv(os.path.join(OUTDIR, "CBT_AUC_model_selection_AIC_BIC.csv"), index=False)

    # fit final model
    res, fml, analysis_df = fit_final_model(metric_df, ms, ycol=ycol)

    with open(os.path.join(OUTDIR, "CBT_AUC_final_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Final model formula:\n")
        f.write(fml + "\n\n")
        f.write(str(res.summary()))

    # main text table
    main_tbl = build_main_text_table(res, analysis_df, ycol=ycol)
    main_tbl.to_csv(os.path.join(OUTDIR, "CBT_AUC_Table_MainText.csv"), index=False)
    main_tbl.to_string(open(os.path.join(OUTDIR, "CBT_AUC_Table_MainText.txt"), "w", encoding="utf-8"), index=False)

    # plot (model-consistent summaries + p-values)
    plot_paired(analysis_df, OUTDIR, ycol=ycol, res=res, cov_source_df=analysis_df)

    print("[DONE] Outputs written to:", OUTDIR)


def main():
    warnings.filterwarnings("ignore")
    run_analysis()


if __name__ == "__main__":
    main()
