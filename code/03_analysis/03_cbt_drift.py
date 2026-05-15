# -*- coding: utf-8 -*-
"""Heat Acclimation — CBT end-to-end pipeline (cleanup + modelling + exports).

Consolidated from:
  - 00_CBTcleanup.py
  - 01_CBTdrift_DD.py
  - 01_CBTdrift_DD_UPDATED_tables_pdf.py

What this script does
---------------------
(1) Raw-data cleanup (optional):
    - Scan *formatted.csv pill logger exports (semicolon CSV)
    - Parse Europe/Berlin datetimes
    - Clip to session window (with per-day/per-participant start overrides)
    - Split by PillID and join Sessions_PartMeta metadata
    - Write per-participant session CSVs

(2) Modelling dataset prep:
    - Load merged CBT table (e.g., 02_AllCBT_flaggedv2.csv)
    - Apply exclusion rules and start-time overrides BEFORE windowing
    - Resample to 1-minute bins (prevents 30-sec logging overweighting)
    - Drop QC-flagged minutes (cbt_flag)
    - Within (participant × scenario) z-score for outlier detection only

(3) Primary estimand:
    - Early window mean  = mean CBT in EARLY_START–EARLY_END
    - Late  window mean  = mean CBT in LATE_START–LATE_END
    - Drift (°C)         = Late − Early
    - DD per participant = HS2 − HS1

(4) Statistics:
    - Mixed effects model on drift:
        drift ~ arm * scenario + covariates + (1|part_id)
    - Model comparison (AIC/BIC) across covariate sets:
        fat_pct vs BMR vs both; optional mens_change
    - Estimated marginal means (HS1/HS2 per arm) and within-arm contrasts
    - Within-arm effect size dz computed on participant DD distribution

(5) Visualisation:
    - Paper-ready paired points plot: HS1 vs HS2 per arm (with jitter + optional ID labels)

(6) Exports:
    - Participant drift table
    - Main-text summary table (HS1, HS2, Δ, CI, p, dz) per arm
    - Model selection table (AIC/BIC)
    - Figures (PNG/PDF)

Notes
-----
- This script uses repository-relative paths configured below.
- The cleanup step (raw pill logger splits) is optional; if you already have the merged CBT file,
  you can set RUN_RAW_CLEANUP = False.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import warnings
from dataclasses import dataclass
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
OUTPUTS = REPO_ROOT / "outputs" / "model_outputs" / "cbt_outputs" / "cbt_drift"

# =========================================================
# CONFIG
# =========================================================

# ---- toggle steps ----
RUN_RAW_CLEANUP = False  # True if you want to regenerate clipped per-participant CBT files

# ---- RAW cleanup paths (only used if RUN_RAW_CLEANUP=True) ----
ROOT_RAW = DATA_RAW / "session_data"
OUT_BASE = DATA_PROCESSED / "CBT_CLIPPED"
META_PSEUD_CSV = DATA_METADATA / "session_meta.csv"
META_PSEUD_SHEET = None  # retained for backwards compatibility; CSV metadata has no sheets
LOCAL_TZ = "Europe/Berlin"

# ---- Analysis paths ----
PATH_CBT = DATA_PROCESSED / "cbt.csv"
PATH_PART = DATA_METADATA / "participant_meta_pseud.csv"

OUTDIR = OUTPUTS / "cbt" / "primary_drift"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- required columns ----
REQ_CBT = ["datetime", "part_id", "condition", "scenario_short", "session_id", "cbt_raw", "cbt_flag"]

# Participant meta: we will *attempt* to use these if present
REQ_PART_BASE = ["part_id", "sex"]
OPTIONAL_PART_COLS = ["fat_pct", "bmr", "mens_change"]

# ---- Arms / scenarios ----
VALID_ARMS = ["FR", "CC"]
ARM_ORDER = ["FR", "CC"]  # plotting/reference

VALID_SCENARIO_SHORT = ["HS_pre", "HS_post"]
SCENARIO_MAP = {"HS_pre": "HS1", "HS_post": "HS2"}
SCEN_ORDER = ["HS1", "HS2"]

# ---- Drift windows (afternoon) ----
EARLY_START = "12:15"
EARLY_END = "12:30"
LATE_START = "16:15"
LATE_END = "16:30"

MIN_VALID_MINUTES_PER_WINDOW = 10
RESAMPLE_TO_MINUTE = True

# ---- QC ----
Z_OUTLIER_THRESH = 4.0

# ---- Post-window completeness QC ----
# Drop participants who do not have BOTH HS1 and HS2 drift values after windowing.
# This avoids asymmetrical PRE-POST datasets caused by missing windows or exclusions.
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
# PRE-MODEL FILTER RULES (REQUESTED)
# =========================================================

# 1) Exclude participant P05 everywhere (dropped out)
EXCLUDE_PARTICIPANTS_ALL = {"P05"}
# EXCLUDE_PARTICIPANTS_ALL = {"P05", "P12", "P13"}

# 2) Exclude P02 only on 2023-06-05
EXCLUDE_PARTICIPANTS_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): {"P02"},
}

# 3) Session start overrides (date-level)
# default session start is assumed to be 09:30 unless overridden
SESSION_START_BY_DATE = {
    pd.Timestamp("2023-06-05").normalize(): "10:40",
    pd.Timestamp("2023-06-06").normalize(): "09:55",
    pd.Timestamp("2023-06-07").normalize(): "09:40",
}

# 4) Participant-specific start overrides (date + part)
SESSION_START_BY_DATE_PART = {
    (pd.Timestamp("2023-06-22").normalize(), "P06"): "10:30",
}

# default session end (kept stable unless you tell me otherwise)
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

    # exclusion logs
    log_rows = []

    # Exclude participants globally
    if EXCLUDE_PARTICIPANTS_ALL:
        m = df["part_id"].astype(str).isin(list(EXCLUDE_PARTICIPANTS_ALL))
        if m.any():
            for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
                log_rows.append({"action": "exclude_all", "part_id": pid, "date": pd.NaT, "reason": "dropped_out"})
        df = df.loc[~m].copy()

    # Exclude participants by date
    for d, pids in EXCLUDE_PARTICIPANTS_BY_DATE.items():
        m = (df["date"] == pd.Timestamp(d).normalize()) & (df["part_id"].astype(str).isin(list(pids)))
        if m.any():
            for pid in sorted(set(df.loc[m, "part_id"].astype(str))):
                log_rows.append({"action": "exclude_date", "part_id": pid, "date": pd.Timestamp(d).normalize(), "reason": "manual_exclusion"})
        df = df.loc[~m].copy()

    # Clip by (date, part_id) start overrides
    # We do this row-wise via computed minute-of-day thresholds.
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
        # summarize clipping by (date, part)
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
# QC: WITHIN-SESSION Z-SCORE (OUTLIER DETECTION ONLY)
# =========================================================

def add_within_session_zscore_qc(df: pd.DataFrame, z_thresh=4.0) -> pd.DataFrame:
    """Compute within (participant×scenario) z-score for outlier detection only."""
    df = df.copy()

    def zscore(g):
        mu = g["cbt_raw"].mean()
        sd = g["cbt_raw"].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            g["cbt_z_qc"] = 0.0
        else:
            g["cbt_z_qc"] = (g["cbt_raw"] - mu) / sd
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
# RAW CLEANUP (OPTIONAL): PILL LOGGER SPLIT + META JOIN
# =========================================================

def read_meta_pseud() -> pd.DataFrame:
    m = pd.read_csv(META_PSEUD_CSV)
    m.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in m.columns]
    need = ["cbt_pill_id", "date", "part_id", "scenario", "condition", "session_id"]
    miss = [c for c in need if c not in m.columns]
    if miss:
        raise KeyError(f"Metadata missing columns: {miss}")
    m["cbt_pill_id"] = m["cbt_pill_id"].astype(str).str.strip()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return m[need]


def read_cbt_semicolon(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    lmap = {c.lower(): c for c in df.columns}

    date_col = lmap.get("date")
    time_col = lmap.get("time")
    pill_col = lmap.get("pillid")

    core_col = None
    for key in ["coretemp", "cbt", "core", "temperature", "temp"]:
        if key in lmap:
            core_col = lmap[key]
            break
        for k, v in lmap.items():
            if re.search(rf"\b{key}\b", k):
                core_col = v
                break
        if core_col:
            break

    if not (date_col and time_col and pill_col and core_col):
        raise KeyError(f"{path.name}: need Date, Time, CoreTemp, PillID. Got {df.columns.tolist()}")

    dt = pd.to_datetime(df[date_col] + " " + df[time_col], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    dt = dt.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")

    out = pd.DataFrame({
        "datetime": dt,
        "pillid": df[pill_col].astype(str).str.strip(),
        "cbt_raw": pd.to_numeric(df[core_col], errors="coerce"),
    })
    out = out.dropna(subset=["datetime", "cbt_raw"]).sort_values("datetime").reset_index(drop=True)
    return out


def yymmdd_to_date(yymmdd: str) -> pd.Timestamp:
    return pd.to_datetime("20" + yymmdd, format="%Y%m%d").normalize()


def find_session_tag(path: Path) -> str | None:
    for ancestor in [path.parent, path.parent.parent, path.parent.parent.parent]:
        if ancestor is None:
            continue
        name = ancestor.name.strip()
        if re.fullmatch(r"\d{6}", name):
            return name
    return None


def nice_filename(part_id, session_date: pd.Timestamp, fallback_tag: str = "UNK") -> str:
    date_str = session_date.strftime("%d%m%Y")
    pid = str(part_id) if pd.notna(part_id) else fallback_tag
    return f"{pid}_{date_str}_CBT.csv"


def session_clip_mask(ts: pd.Series, session_date: pd.Timestamp, part_id: str | None = None) -> pd.Series:
    """Clip to per-day/per-participant start overrides; end fixed at DEFAULT_SESSION_END."""
    pid = str(part_id) if part_id is not None else ""
    start_hhmm = _get_start_hhmm_for_row(session_date, pid)  # date-level unless (date,part) exists
    start = pd.Timestamp.combine(session_date, pd.Timestamp(start_hhmm).time()).tz_localize(LOCAL_TZ)
    end = pd.Timestamp.combine(session_date, pd.Timestamp(DEFAULT_SESSION_END).time()).tz_localize(LOCAL_TZ)
    return (ts >= start) & (ts <= end)


def process_raw_file(path: Path, meta: pd.DataFrame) -> int:
    folder_tag = find_session_tag(path)
    if not folder_tag:
        print(f"[WARN] {path}: cannot find YYMMDD folder in ancestors; skipping.")
        return 0

    session_day = yymmdd_to_date(folder_tag)

    df = read_cbt_semicolon(path)
    # add naive date for join key
    df["date"] = df["datetime"].dt.tz_convert(None).dt.normalize()

    n_written = 0
    for pill_id, chunk in df.groupby("pillid", sort=False):
        day = chunk["date"].iloc[0]
        m = meta[(meta["cbt_pill_id"].astype(str).str.strip() == str(pill_id)) & (meta["date"] == day)]
        if len(m) > 1:
            print(f"[WARN] multiple meta rows for PillID={pill_id} date={day.date()}, taking first.")

        part_id = m["part_id"].iloc[0] if not m.empty else pd.NA
        scenario = m["scenario"].iloc[0] if not m.empty else pd.NA
        condition = m["condition"].iloc[0] if not m.empty else pd.NA
        session_id = m["session_id"].iloc[0] if not m.empty else pd.NA

        # clip using participant-aware override (date+participant)
        mask = session_clip_mask(chunk["datetime"], session_day, str(part_id) if pd.notna(part_id) else None)
        out_df = chunk.loc[mask].copy()
        if out_df.empty:
            continue

        out_df["part_id"] = part_id
        out_df["scenario"] = scenario
        out_df["condition"] = condition
        out_df["session_id"] = session_id

        fname = nice_filename(part_id, session_day, fallback_tag=f"PILL_{re.sub(r'[^A-Za-z0-9]+','',pill_id)}")
        out_path = OUT_BASE / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"[OK] {path.name} -> {out_path.name} (rows={len(out_df)})")
        n_written += 1

    return n_written


def run_raw_cleanup():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    meta = read_meta_pseud()
    files = sorted(ROOT_RAW.rglob("CBT/*_formatted.csv"))
    if not files:
        print("[WARN] no *_formatted.csv files found under CBT/")
        return
    total_chunks = 0
    for p in files:
        try:
            total_chunks += process_raw_file(p, meta)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    print(f"[DONE] wrote {total_chunks} clipped files into {OUT_BASE}")


# =========================================================
# PRIMARY ANALYSIS
# =========================================================

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- CBT ---
    cbt = pd.read_csv(PATH_CBT)
    cbt.columns = [c.strip() for c in cbt.columns]
    require_columns(cbt, REQ_CBT, "CBT")

    # standardise types
    cbt["part_id"] = cbt["part_id"].astype(str)
    cbt["scenario_short"] = cbt["scenario_short"].astype(str)
    cbt["condition"] = cbt["condition"].astype(str)
    cbt = ensure_numeric(cbt, ["cbt_raw"])

    # keep valid arms and scenarios
    cbt = cbt[cbt["condition"].isin(VALID_ARMS)].copy()
    cbt = cbt[cbt["condition"].isin(["FR", "CC"])].copy()
    cbt = cbt[cbt["scenario_short"].isin(VALID_SCENARIO_SHORT)].copy()
    cbt["scenario"] = cbt["scenario_short"].map(SCENARIO_MAP)

    # apply exclusions + clipping rules BEFORE windowing/resampling
    cbt, filt_log = apply_exclusions_and_session_clipping(cbt)

    # apply QC flag filter
    # Keep ONLY rows flagged as OK.
    # Supported conventions:
    #   - cbt_flag == "ok" (string)
    #   - cbt_flag == 0 (numeric)
    # If cbt_flag is missing/NA everywhere, keep all rows.
    if "cbt_flag" in cbt.columns:
        cbt = cbt.copy()
        flag = cbt["cbt_flag"]
        if flag.dtype == object:
            flag_norm = flag.astype(str).str.strip().str.lower()
            ok_mask = flag_norm.eq("ok") | flag_norm.eq("0") | flag_norm.eq("nan")
            # Treat real NaNs as allowed only if the entire column is NA
            if flag.isna().all():
                ok_mask = pd.Series([True]*len(cbt), index=cbt.index)
            else:
                ok_mask = ok_mask | flag.isna()
            cbt = cbt.loc[ok_mask].copy()
        else:
            flag_num = pd.to_numeric(flag, errors="coerce")
            if flag_num.isna().all():
                pass
            else:
                cbt = cbt.loc[flag_num == 0].copy()

    # --- participant meta ---
    part = pd.read_csv(PATH_PART)
    part.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in part.columns]
    require_columns(part, REQ_PART_BASE, "Participants")

    # harmonise key
    part["part_id"] = part["part_id"].astype(str)

    # keep only needed columns
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

    # build covariate variants (centered)
    if "fat_pct" in df.columns:
        df["fat_pct_c"] = df["fat_pct"] - df["fat_pct"].mean(skipna=True)
    if "bmr" in df.columns:
        df["bmr_c"] = df["bmr"] - df["bmr"].mean(skipna=True)

    # mens_change: apply only to females via interaction; set missing/males to 0 to avoid NA drop
    if "mens_change" in df.columns:
        mc = df["mens_change"].copy()
        mc = mc.fillna(0.0)
        df["mens_change_c"] = mc - mc.mean(skipna=True)
        df["mens_change_f"] = df["mens_change_c"] * df["sex_c"].fillna(0.0)

    return df, part, filt_log


def resample_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if not RESAMPLE_TO_MINUTE:
        return df.copy()

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

    return agg


def compute_drift_table(df_min: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return drift per participant×arm×scenario and an exclusion table for insufficient data."""
    d = df_min.copy()

    # QC outliers (within participant×scenario)
    d = d.rename(columns={"condition": "arm"})
    d = add_within_session_zscore_qc(d.rename(columns={"arm": "condition"}), z_thresh=Z_OUTLIER_THRESH)
    d = d.rename(columns={"condition": "arm"})
    d = d[~d["is_outlier"]].copy()

    d["is_early"] = in_window(d["minute_of_day"], EARLY_START, EARLY_END)
    d["is_late"] = in_window(d["minute_of_day"], LATE_START, LATE_END)

    # window means
    gb = ["part_id", "arm", "scenario"]

    early = d[d["is_early"]].groupby(gb).agg(
        n_early=("cbt_raw", "size"),
        early_mean=("cbt_raw", "mean"),
    ).reset_index()

    late = d[d["is_late"]].groupby(gb).agg(
        n_late=("cbt_raw", "size"),
        late_mean=("cbt_raw", "mean"),
    ).reset_index()

    drift = early.merge(late, on=gb, how="outer")
    drift["drift"] = drift["late_mean"] - drift["early_mean"]

    # attach covariates (first available from df_min)
    cov_cols = [c for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"] if c in df_min.columns]
    cov = df_min.groupby(["part_id"], as_index=False)[cov_cols].first() if cov_cols else df_min[["part_id"]].drop_duplicates()
    drift = drift.merge(cov, on="part_id", how="left")

    # exclusions due to insufficient minutes
    excl = drift[(drift["n_early"].fillna(0) < MIN_VALID_MINUTES_PER_WINDOW) | (drift["n_late"].fillna(0) < MIN_VALID_MINUTES_PER_WINDOW)].copy()
    excl["reason"] = (
        "min_minutes_fail(" +
        "n_early=" + excl["n_early"].fillna(0).astype(int).astype(str) +
        ",n_late=" + excl["n_late"].fillna(0).astype(int).astype(str) +
        ")"
    )

    drift_ok = drift.drop(excl.index).copy()

    # enforce ordering
    drift_ok["arm"] = pd.Categorical(drift_ok["arm"], categories=ARM_ORDER, ordered=True)
    drift_ok["scenario"] = pd.Categorical(drift_ok["scenario"], categories=SCEN_ORDER, ordered=True)

    return drift_ok, excl


def model_selection(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare covariate sets via AIC/BIC for the CBT drift model.

    Notes:
    - Fits MixedLM with participant random intercept (groups=part_id).
    - Tries multiple optimizers; if all fail, records the last error for that model.
    - Drops rows with missing variables required by each formula.
    """
    base_terms = "arm * scenario"

    # candidate covariates (include only if available and sufficiently non-missing)
    candidates = []
    if "fat_pct_c" in drift_df.columns and drift_df["fat_pct_c"].notna().sum() >= 10:
        candidates.append("fat_pct_c")
    if "bmr_c" in drift_df.columns and drift_df["bmr_c"].notna().sum() >= 10:
        candidates.append("bmr_c")
    if "mens_change_f" in drift_df.columns and drift_df["mens_change_f"].notna().sum() >= 10:
        candidates.append("mens_change_f")

    # always include sex_c if available
    always = []
    if "sex_c" in drift_df.columns and drift_df["sex_c"].notna().sum() >= 10:
        always.append("sex_c")

    def _terms_str(terms):
        if not terms:
            return ""
        return " + " + " + ".join(terms)

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

    # build model formulas
    formulas = []
    formulas.append(("M0_base", f"drift ~ {base_terms}{_terms_str(always)}"))

    # fat-only / bmr-only cases
    if "fat_pct_c" in candidates and "bmr_c" not in candidates:
        formulas.append(("M1_plus_fat", f"drift ~ {base_terms}{_terms_str(always + ['fat_pct_c'])}"))
    if "bmr_c" in candidates and "fat_pct_c" not in candidates:
        formulas.append(("M1_plus_bmr", f"drift ~ {base_terms}{_terms_str(always + ['bmr_c'])}"))

    # both available -> test fat, bmr, and both
    if "fat_pct_c" in candidates and "bmr_c" in candidates:
        formulas.append(("M1_plus_fat", f"drift ~ {base_terms}{_terms_str(always + ['fat_pct_c'])}"))
        formulas.append(("M2_plus_bmr", f"drift ~ {base_terms}{_terms_str(always + ['bmr_c'])}"))
        formulas.append(("M3_plus_fat_bmr", f"drift ~ {base_terms}{_terms_str(always + ['fat_pct_c', 'bmr_c'])}"))

    # menstrual term: add on top of whatever is available (fat and/or bmr) + always terms
    if "mens_change_f" in candidates:
        extra_covs = [c for c in ["fat_pct_c", "bmr_c"] if (c in drift_df.columns and c in candidates)]
        formulas.append(("M4_plus_mens", f"drift ~ {base_terms}{_terms_str(always + extra_covs + ['mens_change_f'])}"))

    rows = []
    for name, fml in formulas:
        d = drift_df.copy()

        # Extract variable-like tokens and keep only those that exist as columns
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", fml)
        ignore = {"drift", "arm", "scenario"}
        vars_needed = [t for t in tokens if t not in ignore and t in d.columns]

        use_cols = list(dict.fromkeys(["drift", "part_id", "arm", "scenario"] + vars_needed))
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

def fit_final_model(drift_df: pd.DataFrame, model_table: pd.DataFrame):
    """Pick best (lowest AIC) successful model; fall back to base."""
    pick = model_table.dropna(subset=["aic"]).sort_values("aic").head(1)
    if pick.empty:
        fml = "drift ~ arm * scenario"
    else:
        fml = str(pick.iloc[0]["formula"])

    # build analysis df with required vars
    vars_needed = [v for v in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", fml) if v not in {"drift"}]
    use_cols = list(set(["drift", "part_id"] + vars_needed))
    d = drift_df[use_cols].dropna().copy()

    last_err = None
    for method in ["lbfgs", "powell", "cg", "nm"]:
        try:
            res = smf.mixedlm(fml, d, groups=d["part_id"]).fit(reml=True, method=method, maxiter=2000, disp=False)
            return res, fml, d
        except Exception as e:
            last_err = e
            continue
    raise last_err

    return res, fml, d


def build_main_text_table(res, drift_analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Create main-text table: HS1/HS2 adjusted means + within-arm Δ + dz."""
    # compute EMMs for each arm × scenario at mean covariates
    cov_means = {}
    for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"]:
        if c in drift_analysis_df.columns:
            cov_means[c] = float(drift_analysis_df[c].mean())

    rows = []

    # helper to get mean/SE for a given arm/scenario combo
    def emm(arm, scen):
        nd = {"arm": arm, "scenario": scen}
        nd.update(cov_means)
        new_df = pd.DataFrame([nd])
        cvec = build_design_row_from_res(res, new_df)
        mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, cvec)
        return mean, se, lo, hi

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
        return mean, se, tval, p, lo, hi

    # dz: computed on participant DD (HS2-HS1) within arm
    drift_wide = drift_analysis_df.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()

    for arm in ARM_ORDER:
        m1, se1, lo1, hi1 = emm(arm, "HS1")
        m2, se2, lo2, hi2 = emm(arm, "HS2")
        dmean, dse, dt, dp, dlo, dhi = delta_within_arm(arm)

        # dz based on individual DD distribution
        dd = drift_wide.loc[drift_wide["arm"] == arm].copy()
        if "HS1" in dd.columns and "HS2" in dd.columns:
            dd["DD"] = dd["HS2"] - dd["HS1"]
            dd_vals = dd["DD"].dropna().values.astype(float)
            dz = float(np.mean(dd_vals) / np.std(dd_vals, ddof=1)) if dd_vals.size >= 2 and np.std(dd_vals, ddof=1) > 0 else np.nan
            n_dd = int(dd_vals.size)
        else:
            dz = np.nan
            n_dd = 0

        rows.append({
            "arm": arm,
            "n_participants": int(drift_analysis_df.loc[drift_analysis_df["arm"] == arm, "part_id"].nunique()),
            "HS1_adj_mean": m1,
            "HS1_ci_lo": lo1,
            "HS1_ci_hi": hi1,
            "HS2_adj_mean": m2,
            "HS2_ci_lo": lo2,
            "HS2_ci_hi": hi2,
            "delta_HS2_minus_HS1": dmean,
            "delta_ci_lo": dlo,
            "delta_ci_hi": dhi,
            "p_within_arm": dp,
            "dz_within_arm": dz,
            "n_dd": n_dd,
        })

    out = pd.DataFrame(rows)
    return out


def plot_paired(drift_df: pd.DataFrame, outdir: str, res=None, cov_source_df: pd.DataFrame | None = None):
    """Paired points plot HS1/HS2 per arm.

    - Participant points/lines: raw drift values
    - Summary markers + 95% CI: model-adjusted EMMs if res is provided; else raw mean CI
    - Bracket p-values: model-based within-arm HS2-HS1 contrast if res is provided; else paired t-test
    - dz: computed on participant DD distribution within arm (same as build_main_text_table)
    """

    d = drift_df.copy()
    d = d.dropna(subset=["drift", "arm", "scenario", "part_id"]).copy()

    # enforce ordering
    d["arm"] = pd.Categorical(d["arm"], categories=ARM_ORDER, ordered=True)
    d["scenario"] = pd.Categorical(d["scenario"], categories=SCEN_ORDER, ordered=True)

    # numeric x positions
    x_arm = {arm: i for i, arm in enumerate(ARM_ORDER)}
    x_scen = {"HS1": -0.15, "HS2": 0.15}

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    # -------------------------
    # Participant points (raw)
    # -------------------------
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
                dd["drift"].values,
                s=POINT_SIZE,
                marker=marker,
                facecolor="black",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
                zorder=2,
            )

            if LABEL_PARTICIPANTS:
                for xi, yi, pid in zip(xs, dd["drift"].values, dd["part_id"].astype(str).values):
                    ax.text(xi, yi, pid, fontsize=LABEL_FONTSIZE, ha="left", va="center", color="black")

    # pair lines (raw)
    if PAIR_LINES:
        wide = d.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()
        for _, r in wide.iterrows():
            arm = r["arm"]
            if arm not in x_arm:
                continue
            y1 = r.get("HS1", np.nan)
            y2 = r.get("HS2", np.nan)
            if not (np.isfinite(y1) and np.isfinite(y2)):
                continue
            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, alpha=0.6, zorder=1)

    # ---------------------------------------
    # Summary markers + CI (MODEL if possible)
    # ---------------------------------------
    use_model = res is not None

    if use_model:
        # covariate means for EMM evaluation
        if cov_source_df is None:
            cov_source_df = d

        cov_means = {}
        for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"]:
            if c in cov_source_df.columns:
                cov_means[c] = float(cov_source_df[c].mean())

        def emm_mean_ci(arm, scen):
            nd = {"arm": arm, "scenario": scen}
            nd.update(cov_means)
            new_df = pd.DataFrame([nd])
            cvec = build_design_row_from_res(res, new_df)
            mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, cvec)
            return mean, lo, hi

        summary_rows = []
        for arm in ARM_ORDER:
            for scen in SCEN_ORDER:
                m, lo, hi = emm_mean_ci(arm, scen)
                summary_rows.append({"arm": arm, "scenario": scen, "mean": m, "ci_lo": lo, "ci_hi": hi})
        summ = pd.DataFrame(summary_rows)

    else:
        # fallback: raw mean ± 1.96*SE
        summ = d.groupby(["arm", "scenario"]).agg(
            n=("drift", "count"),
            mean=("drift", "mean"),
            sd=("drift", "std"),
        ).reset_index()
        summ["se"] = summ["sd"] / np.sqrt(summ["n"].clip(lower=1))
        summ["ci_lo"] = summ["mean"] - 1.96 * summ["se"]
        summ["ci_hi"] = summ["mean"] + 1.96 * summ["se"]

    # plot summary markers/bars
    for _, r in summ.iterrows():
        arm, scen = r["arm"], r["scenario"]
        if arm not in x_arm or scen not in x_scen:
            continue
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

    # ----------------------------------------------------
    # Brackets: MODEL within-arm contrast if possible
    # ----------------------------------------------------
    try:
        y_min, y_max = ax.get_ylim()
        y_span = max(1e-9, (y_max - y_min))
        y_pad = 0.06 * y_span

        wide_all = d.pivot_table(index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first").reset_index()

        # dz from DD distribution (same logic as build_main_text_table)
        def dz_from_dd(arm):
            ww = wide_all[wide_all["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any")
            if ww.shape[0] < 2:
                return np.nan
            diff = (ww["HS2"] - ww["HS1"]).to_numpy(dtype=float)
            sd = float(np.std(diff, ddof=1))
            return float(np.mean(diff) / sd) if sd > 0 else np.nan

        if use_model:
            # model-based p for HS2-HS1 within each arm
            # reuse same cov_means as EMMs
            if cov_source_df is None:
                cov_source_df = d

            cov_means = {}
            for c in ["sex_c", "fat_pct_c", "bmr_c", "mens_change_f"]:
                if c in cov_source_df.columns:
                    cov_means[c] = float(cov_source_df[c].mean())

            def p_within_arm_model(arm):
                nd1 = {"arm": arm, "scenario": "HS2"}; nd1.update(cov_means)
                nd0 = {"arm": arm, "scenario": "HS1"}; nd0.update(cov_means)
                c1 = build_design_row_from_res(res, pd.DataFrame([nd1]))
                c0 = build_design_row_from_res(res, pd.DataFrame([nd0]))
                c = c1 - c0
                mean, se, tval, p, lo, hi = estimate_fe_linear_combo(res, c)
                return float(p)

        for arm in ARM_ORDER:
            ww = wide_all[wide_all["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any")
            if ww.shape[0] < 3:
                continue

            # y placement based on raw points to avoid overlap
            pre = ww["HS1"].to_numpy(dtype=float)
            post = ww["HS2"].to_numpy(dtype=float)
            y = max(np.nanmax(pre), np.nanmax(post)) + y_pad

            dz = dz_from_dd(arm)

            if use_model:
                p_val = p_within_arm_model(arm)
            else:
                # fallback paired t-test
                t_stat, p_val = stats.ttest_rel(post, pre, nan_policy="omit")
                p_val = float(p_val)

            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]

            ax.plot([x1, x1, x2, x2], [y - 0.25*y_pad, y, y, y - 0.25*y_pad],
                    color="black", linewidth=1.0, zorder=5)
            ax.text((x1 + x2) / 2, y + 0.1*y_pad, f"p={p_val:.3f}, dz={dz:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")

    except Exception:
        pass

    ax.axhline(0, color="black", linewidth=AXH0_LW, alpha=0.7)
    ax.set_xticks([x_arm[a] for a in ARM_ORDER])
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel("CBT drift (Late − Early), °C")
    ax.set_title("CBT drift by arm and scenario (HS1 vs HS2)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=6, label='PRE (HS1)'),
        Line2D([0], [0], marker='s', color='black', linestyle='None', markersize=6, label='POST (HS2)'),
    ]
    ax.legend(handles=handles, frameon=False, loc='best')

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(os.path.join(outdir, "CBT_drift_paired.png"), dpi=300)
    if SAVE_PDF:
        fig.savefig(os.path.join(outdir, "CBT_drift_paired.pdf"), format="pdf")

    plt.close(fig)


# =========================================================
# POST-WINDOW QC: DROP ASYMMETRICAL PARTICIPANTS
# =========================================================

def drop_asymmetrical_participants(drift_df: pd.DataFrame, excl_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop participants who are missing HS1 or HS2 after windowing.

    Parameters
    ----------
    drift_df : DataFrame
        Participant x arm x scenario drift table (already passed window requirements).
    excl_df : DataFrame
        Existing exclusion table (e.g., insufficient minutes).

    Returns
    -------
    drift_df_kept, excl_df_updated, dropped_ids
    """
    if drift_df.empty:
        return drift_df, excl_df, []

    # Count how many distinct scenarios each participant has
    n_scen = drift_df.groupby('part_id')['scenario'].nunique()
    dropped = sorted(n_scen[n_scen < REQUIRED_SCENARIOS_PER_PART].index.astype(str).tolist())
    if not dropped:
        return drift_df, excl_df, []

    # Add to exclusion table for transparency
    add_rows = []
    for pid in dropped:
        have = drift_df.loc[drift_df['part_id'] == pid, 'scenario'].astype(str).unique().tolist()
        add_rows.append({
            'part_id': pid,
            'arm': drift_df.loc[drift_df['part_id'] == pid, 'arm'].astype(str).iloc[0] if (drift_df['part_id'] == pid).any() else '',
            'scenario': 'NA',
            'n_early': np.nan,
            'early_mean': np.nan,
            'n_late': np.nan,
            'late_mean': np.nan,
            'drift': np.nan,
            'reason': 'asymmetrical_missing_scenario(have=' + ','.join(sorted(have)) + ')',
        })
    excl_add = pd.DataFrame(add_rows)
    excl_df2 = pd.concat([excl_df.copy(), excl_add], ignore_index=True, sort=False)

    drift_kept = drift_df.loc[~drift_df['part_id'].astype(str).isin(dropped)].copy()
    return drift_kept, excl_df2, dropped


def run_analysis():
    df, part, filt_log = load_and_prepare()

    # exports: filter log
    filt_log_path = os.path.join(OUTDIR, "CBT_filtering_log.csv")
    filt_log.to_csv(filt_log_path, index=False)

    # minute resample
    df_min = resample_to_minutes(df)

    # compute drift
    drift_df, excl_df = compute_drift_table(df_min)

    # Drop participants with missing HS1 or HS2 after windowing (optional)
    dropped_asym = []
    if DROP_ASYMMETRICAL_PARTICIPANTS:
        drift_df, excl_df, dropped_asym = drop_asymmetrical_participants(drift_df, excl_df)
        if dropped_asym:
            print('Dropped asymmetrical participants (missing HS1 or HS2):', dropped_asym)

    # --- DEBUG: model input visibility (Spyder variable explorer) ---
    global drift_analysis_df_debug
    drift_analysis_df_debug = drift_df.copy()

    print("\n=== DRIFT TABLE (after filtering + window requirements) ===")
    print("Rows (part×scenario):", drift_df.shape[0])
    print("Participants:", drift_df["part_id"].nunique())
    print("Arms per participant:", drift_df.groupby("arm")["part_id"].nunique().to_dict())
    print("Counts arm×scenario:\n", pd.crosstab(drift_df["arm"], drift_df["scenario"]))

    drift_path = os.path.join(OUTDIR, "CBT_drift_participant_level.csv")
    drift_df.to_csv(drift_path, index=False)

    excl_path = os.path.join(OUTDIR, "CBT_drift_exclusions_insufficient_minutes.csv")
    excl_df.to_csv(excl_path, index=False)

    # model selection
    ms = model_selection(drift_df)
    ms.to_csv(os.path.join(OUTDIR, "CBT_model_selection_AIC_BIC.csv"), index=False)

    # fit final model
    res, fml, drift_analysis_df = fit_final_model(drift_df, ms)

    with open(os.path.join(OUTDIR, "CBT_final_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Final model formula:\n")
        f.write(fml + "\n\n")
        f.write(str(res.summary()))

    # main text table
    main_tbl = build_main_text_table(res, drift_analysis_df)
    main_tbl.to_csv(os.path.join(OUTDIR, "CBT_Table_MainText.csv"), index=False)
    main_tbl.to_string(open(os.path.join(OUTDIR, "CBT_Table_MainText.txt"), "w", encoding="utf-8"), index=False)

    # plot
    # plot_paired(drift_analysis_df, OUTDIR)
    plot_paired(drift_analysis_df, OUTDIR, res=res, cov_source_df=drift_analysis_df)


    print("[DONE] Outputs written to:", OUTDIR)


def main():
    warnings.filterwarnings("ignore")
    if RUN_RAW_CLEANUP:
        run_raw_cleanup()
    run_analysis()


if __name__ == "__main__":
    main()
