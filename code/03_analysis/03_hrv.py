# -*- coding: utf-8 -*-
"""
Heat Acclimation — HR + HRV drift pipeline (HR, lnRMSSD, lnHF)
Compatible with your 02_AllHRV_withSed.csv schema:
  - HR: bpm_mean
  - RMSSD: hrv_rmssd (ms) -> lnRMSSD = ln(hrv_rmssd)
  - HF power: hrv_hf (ms^2 or power units) -> lnHF = ln(max(hrv_hf, HF_FLOOR))

Outputs:
  HRV_filtering_log.csv
  HR_* (drift table, exclusions, model selection, final model summary, main text table, plots)
  lnRMSSD_* (same)
  lnHF_* (same)
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import t as t_dist



# =========================================================
# REPOSITORY PATHS
# =========================================================
# This file is intended to live in: <repo>/code/03_analysis/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_METADATA = REPO_ROOT / "data" / "metadata"
OUTPUTS = REPO_ROOT / "outputs" / "model_outputs" / "hrv_outputs"

# =========================================================
# CONFIG
# =========================================================

PATH_HRV = DATA_PROCESSED / "hrv.csv"
PATH_PART = DATA_METADATA / "participant_meta_pseud.csv"

OUTDIR = OUTPUTS / "hrv" / "primary_drift"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Arms/scenarios
VALID_ARMS = ["FR", "CC"]
ARM_ORDER = ["FR", "CC"]

# In your HRV file, scenario is likely HS_pre/HS_post or HS1/HS2. We'll normalize.
SCEN_ORDER = ["HS1", "HS2"]
SCENARIO_MAP = {"HS_pre": "HS1", "HS_post": "HS2", "HS1": "HS1", "HS2": "HS2"}

# Drift windows (shifted earlier vs CBT)
EARLY_START = "11:45"
EARLY_END   = "12:15"
LATE_START  = "15:45"
LATE_END    = "16:15"
MIN_VALID_MINUTES_PER_WINDOW = 10

# Exclusions / clipping (match CBT)
EXCLUDE_PARTICIPANTS_ALL = {"P05"}  # dropped out
EXCLUDE_PARTICIPANTS_BY_DATE = {pd.Timestamp("2023-06-05").normalize(): {"P02"}}

SESSION_START_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): "10:40",
    pd.Timestamp("2023-06-06").normalize(): "09:55",
    pd.Timestamp("2023-06-07").normalize(): "09:40",
}
SESSION_START_BY_DATE_PART = {(pd.Timestamp("2023-06-22").normalize(), "P06"): "10:30"}
DEFAULT_SESSION_END = "16:30"

# Sedentary filtering
REQUIRE_SEDENTARY_TRUE = True

# Outlier filter within (part×scenario) per metric
Z_OUTLIER_THRESH = 3.0

# Drop participants missing HS1 or HS2 after windowing
DROP_ASYMMETRICAL_PARTICIPANTS = True
REQUIRED_SCENARIOS_PER_PART = 2

# P04 QC
CHECK_P04 = True
EXCLUDE_P04_IF_FAILS_QC = True
P04_ID = "P04"

HR_PLAUSIBLE_RANGE = (40, 200)           # bpm
LNRMSSD_PLAUSIBLE_RANGE = (-1.0, 10.0)   # ln(ms)
LNHF_PLAUSIBLE_RANGE = (-30.0, 30.0)     # ln(power); broad, just catches explosions
HF_FLOOR = 1e-6                          # to avoid -inf when HF power hits 0

# Plot
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
# HELPERS
# =========================================================

def clock_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def in_window(mins: pd.Series, start_hhmm: str, end_hhmm: str) -> pd.Series:
    a = clock_to_minutes(start_hhmm)
    b = clock_to_minutes(end_hhmm)
    return (mins >= a) & (mins < b)

def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()
    df["date"] = df["datetime"].dt.normalize()
    df["minute_floor"] = df["datetime"].dt.floor("min")
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    return df

def normalize_bool_series(x: pd.Series) -> pd.Series:
    s = x.copy()
    if s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        is_true = ss.isin(["true", "1", "yes", "y", "t"])
        return is_true.fillna(False)
    sn = pd.to_numeric(s, errors="coerce")
    return (sn == 1).fillna(False)

def robust_ci_from_t(mean, se, df_resid, alpha=0.05):
    if not np.isfinite(se) or se <= 0 or not np.isfinite(df_resid) or df_resid <= 0:
        return np.nan, np.nan
    tcrit = float(t_dist.ppf(1 - alpha / 2, df_resid))
    return mean - tcrit * se, mean + tcrit * se

def _get_start_hhmm_for_row(date_val: pd.Timestamp, part_id: str) -> str:
    key = (pd.Timestamp(date_val).normalize(), str(part_id))
    if key in SESSION_START_BY_DATE_PART:
        return SESSION_START_BY_DATE_PART[key]
    d = pd.Timestamp(date_val).normalize()
    return SESSION_START_BY_DATE.get(d, "09:30")


# =========================================================
# EXCLUSIONS + CLIPPING
# =========================================================

def apply_exclusions_and_session_clipping(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_time_fields(df)

    log_rows = []

    # Exclude globally
    if EXCLUDE_PARTICIPANTS_ALL:
        m = df["part_id"].astype(str).isin(list(EXCLUDE_PARTICIPANTS_ALL))
        for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
            log_rows.append({"action": "exclude_all", "part_id": pid, "date": pd.NaT, "reason": "dropped_out", "n_rows": int(m.sum())})
        df = df.loc[~m].copy()

    # Exclude by date
    for d, pids in EXCLUDE_PARTICIPANTS_BY_DATE.items():
        m = (df["date"] == pd.Timestamp(d).normalize()) & (df["part_id"].astype(str).isin(list(pids)))
        for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
            log_rows.append({"action": "exclude_date", "part_id": pid, "date": pd.Timestamp(d).normalize(), "reason": "manual_exclusion", "n_rows": int(m.sum())})
        df = df.loc[~m].copy()

    # Clip rows outside (start_override .. DEFAULT_SESSION_END)
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
        g = clipped.groupby(["date", "part_id"]).size().reset_index(name="n_rows")
        for _, r in g.iterrows():
            log_rows.append({
                "action": "clip_before_start",
                "part_id": str(r["part_id"]),
                "date": pd.Timestamp(r["date"]).normalize(),
                "reason": f"start_override={_get_start_hhmm_for_row(pd.Timestamp(r['date']).normalize(), str(r['part_id']))}",
                "n_rows": int(r["n_rows"]),
            })

    df = df.loc[keep_mask].copy()
    log = pd.DataFrame(log_rows, columns=["action", "part_id", "date", "reason", "n_rows"])
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
    mean = float((c.T @ beta)[0, 0])
    var = float((c.T @ V @ c)[0, 0])
    se = float(np.sqrt(max(var, 0.0)))

    df_resid = float(getattr(res, "df_resid", np.nan))
    if np.isfinite(df_resid) and df_resid > 0 and se > 0:
        tval = mean / se
        p = 2 * (1 - t_dist.cdf(abs(tval), df_resid))
        lo, hi = robust_ci_from_t(mean, se, df_resid)
    else:
        p = np.nan
        lo, hi = mean - 1.96 * se, mean + 1.96 * se

    return mean, se, p, lo, hi


# =========================================================
# LOAD / PREP
# =========================================================

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PATH_HRV)
    df.columns = [c.strip() for c in df.columns]

    # Canonical fields we will use:
    # datetime, part_id, condition (arm), scenario, sedentary, bpm_mean, hrv_rmssd, hrv_hf
    needed = ["datetime", "part_id", "condition", "scenario", "sedentary", "bpm_mean", "hrv_rmssd", "hrv_hf"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"HRV file missing columns: {missing}\nAvailable: {list(df.columns)}")

    # Types
    df["part_id"] = df["part_id"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["scenario"] = df["scenario"].astype(str)

    df["bpm_mean"] = pd.to_numeric(df["bpm_mean"], errors="coerce")
    df["hrv_rmssd"] = pd.to_numeric(df["hrv_rmssd"], errors="coerce")
    df["hrv_hf"] = pd.to_numeric(df["hrv_hf"], errors="coerce")

    # sedentary filter
    if REQUIRE_SEDENTARY_TRUE:
        df = df.loc[normalize_bool_series(df["sedentary"])].copy()

    # keep arms
    df = df[df["condition"].isin(VALID_ARMS)].copy()
    df["arm"] = df["condition"]

    # normalize scenario to HS1/HS2
    df["scenario"] = df["scenario"].map(SCENARIO_MAP)
    df = df[df["scenario"].isin(SCEN_ORDER)].copy()

    # compute lnRMSSD (guard against non-positive rmssd)
    df["lnrmssd"] = np.where(df["hrv_rmssd"] > 0, np.log(df["hrv_rmssd"]), np.nan)

    # compute lnHF (robust floor to avoid -inf)
    hf = df["hrv_hf"].to_numpy(dtype=float)
    hf_safe = np.where(np.isfinite(hf), np.maximum(hf, HF_FLOOR), np.nan)
    df["lnhf"] = np.log(hf_safe)

    # Apply exclusions + session clipping
    df, filt_log = apply_exclusions_and_session_clipping(df)

    # Participant meta
    part = pd.read_csv(PATH_PART)
    part.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in part.columns]
    if "part_id" not in part.columns or "sex" not in part.columns:
        raise KeyError(f"Participants meta must include part_id and sex. Available: {list(part.columns)}")
    part["part_id"] = part["part_id"].astype(str)
    part["sex"] = part["sex"].astype(str).str.strip().str.upper()
    part["sex_c"] = part["sex"].map({"F": 1, "M": 0, "FEMALE": 1, "MALE": 0}).astype(float)

    if "fat_pct" in part.columns:
        part["fat_pct"] = pd.to_numeric(part["fat_pct"], errors="coerce")
        part["fat_pct_c"] = part["fat_pct"] - part["fat_pct"].mean(skipna=True)

    # Merge
    keep_cols = ["part_id", "sex", "sex_c"] + (["fat_pct", "fat_pct_c"] if "fat_pct_c" in part.columns else [])
    part2 = part[keep_cols].copy()
    df = df.merge(part2, on="part_id", how="left")

    # final time fields
    df = add_time_fields(df)

    # write filtering log
    filt_log.to_csv(os.path.join(OUTDIR, "HRV_filtering_log.csv"), index=False)

    return df, part2


# =========================================================
# QC P04
# =========================================================

def p04_qc(df: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons = []
    if P04_ID not in set(df["part_id"].astype(str)):
        return True, ["P04 not present after initial filters."]

    d = df[df["part_id"].astype(str) == P04_ID].copy()

    # window availability
    d["is_early"] = in_window(d["minute_of_day"], EARLY_START, EARLY_END)
    d["is_late"] = in_window(d["minute_of_day"], LATE_START, LATE_END)
    counts = d.groupby("scenario").agg(n_early=("is_early", "sum"), n_late=("is_late", "sum")).reset_index()

    fail = False
    for scen in SCEN_ORDER:
        row = counts[counts["scenario"] == scen]
        if row.empty:
            fail = True
            reasons.append(f"P04 missing {scen}.")
            continue
        ne, nl = int(row["n_early"].iloc[0]), int(row["n_late"].iloc[0])
        if ne < MIN_VALID_MINUTES_PER_WINDOW or nl < MIN_VALID_MINUTES_PER_WINDOW:
            fail = True
            reasons.append(f"P04 insufficient minutes in {scen} (n_early={ne}, n_late={nl}).")

    # plausibility
    hr_bad = d["bpm_mean"].notna() & ((d["bpm_mean"] < HR_PLAUSIBLE_RANGE[0]) | (d["bpm_mean"] > HR_PLAUSIBLE_RANGE[1]))
    if hr_bad.any():
        fail = True
        reasons.append("P04 has implausible HR values.")

    lr_bad = d["lnrmssd"].notna() & ((d["lnrmssd"] < LNRMSSD_PLAUSIBLE_RANGE[0]) | (d["lnrmssd"] > LNRMSSD_PLAUSIBLE_RANGE[1]))
    if lr_bad.any():
        fail = True
        reasons.append("P04 has implausible lnRMSSD values.")

    lh_bad = d["lnhf"].notna() & ((d["lnhf"] < LNHF_PLAUSIBLE_RANGE[0]) | (d["lnhf"] > LNHF_PLAUSIBLE_RANGE[1]))
    if lh_bad.any():
        fail = True
        reasons.append("P04 has implausible lnHF values.")

    return fail, reasons


# =========================================================
# OUTLIER FILTER
# =========================================================

def zfilter_within_part_scenario(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.copy()
    d = d[d[col].notna()].copy()

    def _z(g):
        mu = g[col].mean()
        sd = g[col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            g["_z"] = 0.0
        else:
            g["_z"] = (g[col] - mu) / sd
        return g

    d = d.groupby(["part_id", "scenario"], group_keys=False).apply(_z)
    return d.loc[d["_z"].abs() <= Z_OUTLIER_THRESH].drop(columns=["_z"], errors="ignore")


# =========================================================
# DRIFT TABLE
# =========================================================

def compute_drift(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = str(metric)

    if metric == "HR":
        val = "bpm_mean"
    elif metric == "lnRMSSD":
        val = "lnrmssd"
    elif metric == "lnHF":
        val = "lnhf"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    d = zfilter_within_part_scenario(df, val)

    d["is_early"] = in_window(d["minute_of_day"], EARLY_START, EARLY_END)
    d["is_late"]  = in_window(d["minute_of_day"], LATE_START, LATE_END)

    gb = ["part_id", "arm", "scenario"]
    early = d[d["is_early"]].groupby(gb).agg(n_early=(val, "size"), early_mean=(val, "mean")).reset_index()
    late  = d[d["is_late"]].groupby(gb).agg(n_late=(val, "size"), late_mean=(val, "mean")).reset_index()

    drift = early.merge(late, on=gb, how="outer")
    drift["drift"] = drift["late_mean"] - drift["early_mean"]

    # attach covariates
    cov_cols = [c for c in ["sex_c", "fat_pct_c"] if c in df.columns]
    cov = df.groupby("part_id", as_index=False)[cov_cols].first() if cov_cols else df[["part_id"]].drop_duplicates()
    drift = drift.merge(cov, on="part_id", how="left")

    excl = drift[
        (drift["n_early"].fillna(0) < MIN_VALID_MINUTES_PER_WINDOW) |
        (drift["n_late"].fillna(0) < MIN_VALID_MINUTES_PER_WINDOW)
    ].copy()
    excl["reason"] = "min_minutes_fail(n_early=" + excl["n_early"].fillna(0).astype(int).astype(str) + ",n_late=" + excl["n_late"].fillna(0).astype(int).astype(str) + ")"

    drift_ok = drift.drop(excl.index).copy()
    drift_ok["arm"] = pd.Categorical(drift_ok["arm"], categories=ARM_ORDER, ordered=True)
    drift_ok["scenario"] = pd.Categorical(drift_ok["scenario"], categories=SCEN_ORDER, ordered=True)

    return drift_ok, excl


def drop_asym(drift_df: pd.DataFrame, excl_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_scen = drift_df.groupby("part_id")["scenario"].nunique()
    drop_ids = sorted(n_scen[n_scen < REQUIRED_SCENARIOS_PER_PART].index.astype(str).tolist())
    if not drop_ids:
        return drift_df, excl_df

    add = []
    for pid in drop_ids:
        have = drift_df.loc[drift_df["part_id"] == pid, "scenario"].astype(str).unique().tolist()
        add.append({"part_id": pid, "arm": drift_df.loc[drift_df["part_id"] == pid, "arm"].astype(str).iloc[0],
                    "scenario": "NA", "n_early": np.nan, "early_mean": np.nan, "n_late": np.nan, "late_mean": np.nan,
                    "drift": np.nan, "reason": "asym_missing(have=" + ",".join(sorted(have)) + ")"})
    excl2 = pd.concat([excl_df, pd.DataFrame(add)], ignore_index=True, sort=False)
    kept = drift_df.loc[~drift_df["part_id"].astype(str).isin(drop_ids)].copy()
    return kept, excl2


# =========================================================
# MODEL SELECTION + FINAL MODEL
# =========================================================

def _fit_multiopt(fml: str, d: pd.DataFrame, reml: bool):
    last = None
    for method in ["lbfgs", "powell", "cg", "nm"]:
        try:
            return smf.mixedlm(fml, d, groups=d["part_id"]).fit(reml=reml, method=method, maxiter=2000, disp=False)
        except Exception as e:
            last = e
    raise last

def model_selection(drift_df: pd.DataFrame) -> pd.DataFrame:
    formulas = [
        ("M0_base", "drift ~ arm * scenario"),
        ("M1_plus_sex", "drift ~ arm * scenario + sex_c"),
        ("M2_plus_sex_fat", "drift ~ arm * scenario + sex_c + fat_pct_c"),
    ]
    rows = []
    for name, fml in formulas:
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", fml)
        use_cols = list(dict.fromkeys(["drift", "part_id"] + [t for t in tokens if t != "drift"]))
        d = drift_df.copy()
        for c in use_cols:
            if c not in d.columns:
                d[c] = np.nan
        d = d[use_cols].dropna().copy()
        n = int(d["part_id"].nunique())
        if n < 3:
            rows.append({"model": name, "formula": fml, "n": n, "aic": np.nan, "bic": np.nan, "error": f"n={n}"})
            continue
        try:
            res = _fit_multiopt(fml, d, reml=False)
            rows.append({"model": name, "formula": fml, "n": n, "aic": float(res.aic), "bic": float(res.bic), "error": ""})
        except Exception as e:
            rows.append({"model": name, "formula": fml, "n": n, "aic": np.nan, "bic": np.nan, "error": str(e)})
    out = pd.DataFrame(rows)
    if out["aic"].notna().any():
        out["delta_aic_vs_best"] = out["aic"] - out["aic"].min()
    else:
        out["delta_aic_vs_best"] = np.nan
    return out.sort_values(["delta_aic_vs_best"], na_position="last")

def fit_final(drift_df: pd.DataFrame, ms: pd.DataFrame):
    pick = ms.dropna(subset=["aic"]).sort_values("aic").head(1)
    fml = "drift ~ arm * scenario + sex_c"
    if not pick.empty:
        fml = str(pick.iloc[0]["formula"])

    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", fml)
    use_cols = list(dict.fromkeys(["drift", "part_id"] + [t for t in tokens if t != "drift"]))
    d = drift_df.copy()
    for c in use_cols:
        if c not in d.columns:
            d[c] = np.nan
    d = d[use_cols].dropna().copy()
    res = _fit_multiopt(fml, d, reml=True)
    return res, fml, d


# =========================================================
# MAIN-TEXT TABLE (CBT STYLE)
# =========================================================

def main_text_table(res, d_model: pd.DataFrame, drift_analysis_df: pd.DataFrame, unit_label: str) -> pd.DataFrame:
    cov_means = {}
    for c in ["sex_c", "fat_pct_c"]:
        if c in d_model.columns:
            cov_means[c] = float(d_model[c].mean())

    def emm(arm, scen):
        nd = {"arm": arm, "scenario": scen}
        nd.update(cov_means)
        cvec = build_design_row_from_res(res, pd.DataFrame([nd]))
        mean, se, p, lo, hi = estimate_fe_linear_combo(res, cvec)
        return mean, lo, hi

    def delta_within(arm):
        nd2 = {"arm": arm, "scenario": "HS2"}; nd2.update(cov_means)
        nd1 = {"arm": arm, "scenario": "HS1"}; nd1.update(cov_means)
        c2 = build_design_row_from_res(res, pd.DataFrame([nd2]))
        c1 = build_design_row_from_res(res, pd.DataFrame([nd1]))
        mean, se, p, lo, hi = estimate_fe_linear_combo(res, c2 - c1)
        return mean, lo, hi, p

    # ΔΔ contrast
    def delta_delta():
        nd_fr2 = {"arm": "FR", "scenario": "HS2"}
        nd_fr1 = {"arm": "FR", "scenario": "HS1"}
        nd_cc2 = {"arm": "CC", "scenario": "HS2"}
        nd_cc1 = {"arm": "CC", "scenario": "HS1"}
        for nd in [nd_fr2, nd_fr1, nd_cc2, nd_cc1]:
            nd.update(cov_means)
        c = (build_design_row_from_res(res, pd.DataFrame([nd_fr2])) - build_design_row_from_res(res, pd.DataFrame([nd_fr1]))) - \
            (build_design_row_from_res(res, pd.DataFrame([nd_cc2])) - build_design_row_from_res(res, pd.DataFrame([nd_cc1])))
        mean, se, p, lo, hi = estimate_fe_linear_combo(res, c)
        return mean, lo, hi, p

    # dz within-arm from participant DD
    wide = drift_analysis_df.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()
    wide["DD"] = wide.get("HS2", np.nan) - wide.get("HS1", np.nan)

    rows = []
    for arm in ARM_ORDER:
        n = int(drift_analysis_df.loc[drift_analysis_df["arm"] == arm, "part_id"].nunique())
        pre, _, _ = emm(arm, "HS1")
        post, _, _ = emm(arm, "HS2")
        dlt, lo, hi, p = delta_within(arm)

        dd = wide.loc[wide["arm"] == arm, "DD"].dropna().to_numpy(dtype=float)
        if dd.size >= 2:
            sd = float(np.std(dd, ddof=1))
            dz = float(np.mean(dd) / sd) if sd > 0 else np.nan
        else:
            dz = np.nan

        rows.append({
            "Arm / Contrast": f"{arm} (n = {n})",
            f"PRE drift ({unit_label})": pre,
            f"POST drift ({unit_label})": post,
            f"Δ POST–PRE ({unit_label})": dlt,
            "95% CI of Δ": f"[{lo:.2f}, {hi:.2f}]",
            "p (paired contrast)": p,
            "Effect size (dz)": dz,
        })

    ddm, ddlo, ddhi, ddp = delta_delta()

    # Cohen's d between arms on DD
    dd_fr = wide.loc[wide["arm"] == "FR", "DD"].dropna().to_numpy(dtype=float)
    dd_cc = wide.loc[wide["arm"] == "CC", "DD"].dropna().to_numpy(dtype=float)
    d_between = np.nan
    if dd_fr.size >= 2 and dd_cc.size >= 2:
        n1, n2 = dd_fr.size, dd_cc.size
        s1 = float(np.std(dd_fr, ddof=1)); s2 = float(np.std(dd_cc, ddof=1))
        sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / max(1, (n1+n2-2)))
        if sp > 0:
            d_between = float((np.mean(dd_fr) - np.mean(dd_cc)) / sp)

    rows.append({
        "Arm / Contrast": "FR – CC (ΔΔ)",
        f"PRE drift ({unit_label})": np.nan,
        f"POST drift ({unit_label})": np.nan,
        f"Δ POST–PRE ({unit_label})": ddm,
        "95% CI of Δ": f"[{ddlo:.2f}, {ddhi:.2f}]",
        "p (paired contrast)": ddp,
        "Effect size (dz)": d_between,  # report as Cohen's d in text
    })

    return pd.DataFrame(rows)


# =========================================================
# PLOT
# =========================================================

def plot_paired(drift_df: pd.DataFrame, ylab: str, title: str, fname: str):
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
            xs = x0 + rng.uniform(-JITTER, JITTER, size=len(dd))
            marker = "o" if scen == "HS1" else "s"
            ax.scatter(xs, dd["drift"].values, s=POINT_SIZE, marker=marker,
                       facecolor="black", edgecolor="black", linewidth=0.6, alpha=0.85)
            if LABEL_PARTICIPANTS:
                for xi, yi, pid in zip(xs, dd["drift"].values, dd["part_id"].astype(str).values):
                    ax.text(xi, yi, pid, fontsize=LABEL_FONTSIZE, ha="left", va="center", color="black")

    if PAIR_LINES:
        wide = d.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()
        for _, r in wide.iterrows():
            arm = r["arm"]
            y1 = r.get("HS1", np.nan); y2 = r.get("HS2", np.nan)
            if not (np.isfinite(y1) and np.isfinite(y2)):
                continue
            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, alpha=0.6)

    for _, r in summ.iterrows():
        arm, scen = r["arm"], r["scenario"]
        x0 = x_arm[arm] + x_scen[scen]
        marker = "o" if scen == "HS1" else "s"
        ax.errorbar([x0], [r["mean"]],
                    yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt=marker, color="black", ecolor="black", elinewidth=ERR_LW,
                    capsize=4, markersize=6, markerfacecolor="black", markeredgecolor="black")

    ax.axhline(0, color="black", linewidth=AXH0_LW, alpha=0.7)
    ax.set_xticks([x_arm[a] for a in ARM_ORDER])
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(os.path.join(OUTDIR, f"{fname}.png"), dpi=300)
    if SAVE_PDF:
        fig.savefig(os.path.join(OUTDIR, f"{fname}.pdf"), format="pdf")
    plt.close(fig)


# =========================================================
# RUN
# =========================================================

def run_metric(df: pd.DataFrame, metric: str):
    drift_df, excl_df = compute_drift(df, metric)
    if DROP_ASYMMETRICAL_PARTICIPANTS:
        drift_df, excl_df = drop_asym(drift_df, excl_df)

    drift_df.to_csv(os.path.join(OUTDIR, f"{metric}_drift_participant_level.csv"), index=False)
    excl_df.to_csv(os.path.join(OUTDIR, f"{metric}_drift_exclusions.csv"), index=False)

    ms = model_selection(drift_df)
    ms.to_csv(os.path.join(OUTDIR, f"{metric}_model_selection_AIC_BIC.csv"), index=False)

    res, fml, d_model = fit_final(drift_df, ms)
    with open(os.path.join(OUTDIR, f"{metric}_final_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Final model formula:\n")
        f.write(fml + "\n\n")
        f.write(str(res.summary()))

    if metric == "HR":
        unit = "bpm"
    elif metric == "lnRMSSD":
        unit = "ln(ms)"
    elif metric == "lnHF":
        unit = "ln(power)"
    else:
        unit = ""

    tbl = main_text_table(res, d_model, drift_df, unit_label=unit)
    tbl.to_csv(os.path.join(OUTDIR, f"{metric}_Table_MainText.csv"), index=False)
    tbl.to_string(open(os.path.join(OUTDIR, f"{metric}_Table_MainText.txt"), "w", encoding="utf-8"), index=False)

    if metric == "HR":
        ylab = "HR drift (Late − Early), bpm"
    elif metric == "lnRMSSD":
        ylab = "lnRMSSD drift (Late − Early), ln(ms)"
    elif metric == "lnHF":
        ylab = "lnHF drift (Late − Early), ln(power)"
    else:
        ylab = "Drift (Late − Early)"

    title = f"{metric} drift by arm and scenario (HS1 vs HS2)\nWindows: {EARLY_START}–{EARLY_END} and {LATE_START}–{LATE_END} (sedentary only)"
    plot_paired(drift_df, ylab=ylab, title=title, fname=f"{metric}_paired")

    print(f"\n=== {metric} ===")
    print("Participants:", drift_df["part_id"].nunique())
    print(pd.crosstab(drift_df["arm"], drift_df["scenario"]))


def run_analysis():
    df, part = load_and_prepare()

    # P04 QC
    if CHECK_P04:
        fail, reasons = p04_qc(df)
        if fail and EXCLUDE_P04_IF_FAILS_QC:
            df = df[df["part_id"].astype(str) != P04_ID].copy()
            print("[QC] Excluded P04:", reasons)
        else:
            print("[QC] P04:", ("FAIL" if fail else "PASS"), reasons)

    run_metric(df, "HR")
    run_metric(df, "lnRMSSD")
    run_metric(df, "lnHF")

    print("\n[DONE] Outputs:", OUTDIR)


def main():
    warnings.filterwarnings("ignore")
    run_analysis()

if __name__ == "__main__":
    main()
