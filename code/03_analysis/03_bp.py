# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:49:04 2026


Heat Acclimation — Blood Pressure drift pipeline (SBP/DBP/MAP/PP), aligned to SkinTemp pipeline.

Primary drift definition (recommended):
- Early (baseline): mean(measurement_timestep 3 and 4)  [11:30, 12:30]
- Late  (endpoint): mean(measurement_timestep 7 and 8)  [15:30, 16:30]
- Drift = Late − Early

Also writes sensitivity drift tables:
- drift_3_to_7: timestep 7 − timestep 3
- drift_4_to_8: timestep 8 − timestep 4

Models:
- Final (REML): drift ~ arm * scenario + sex_c   with (1|part_id)
- Model-selection (ML): AIC/BIC across:
    M0: drift ~ arm * scenario
    M1: drift ~ arm * scenario + sex_c
    M2: drift ~ arm * scenario + sex_c + fat_pct
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
OUTPUTS = REPO_ROOT / "outputs" / "model_outputs" / "bp_outputs"

# =========================================================
# CONFIG (match your SkinTemp style)
# =========================================================

PATH_BP = DATA_PROCESSED / "bp.csv"

PATH_PART = DATA_METADATA / "participant_meta_pseud.csv"

OUTDIR = OUTPUTS / "blood_pressure" / "primary_drift"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Arms + scenarios (keep consistent with your other pipelines)
VALID_ARMS = ["FR", "CC"]
ARM_ORDER = ["FR", "CC"]

VALID_SCENARIOS = ["HS1", "HS2"]
SCEN_ORDER = ["HS1", "HS2"]

# Drift windows using measurement_timestep
EARLY_STEPS = [3, 4]   # 11:30, 12:30 (pre-lunch)
LATE_STEPS  = [7, 8]   # 15:30, 16:30 (late-session)

SENSITIVITY_DEFS = {
    "drift_3_to_7": {"early": [3], "late": [7]},
    "drift_4_to_8": {"early": [4], "late": [8]},
}

MIN_MEAS_PER_WINDOW = 1  # BP sparse by design

# Plot controls (mirror SkinTemp; set jitter=0 for BP)
PAIR_LINES = True
JITTER = 0.0
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

def robust_ci_from_t(mean, se, df_resid, alpha=0.05):
    if not np.isfinite(se) or se <= 0 or not np.isfinite(df_resid) or df_resid <= 0:
        return np.nan, np.nan
    tcrit = float(t_dist.ppf(1 - alpha / 2, df_resid))
    return mean - tcrit * se, mean + tcrit * se


# =========================================================
# MIXEDLM CONTRAST UTILITIES (same pattern as SkinTemp)
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

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    bp = pd.read_csv(PATH_BP)
    bp.columns = [c.strip() for c in bp.columns]

    # Expect at least these; adjust names here ONLY if your CSV uses different headers
    REQ_BP = ["part_id", "condition", "scenario", "measurement_timestep", "sbp", "dbp"]
    # If your file uses bp_sys / bp_dia instead, uncomment mapping below.
    if "sbp" not in bp.columns and "bp_sys" in bp.columns:
        bp = bp.rename(columns={"bp_sys": "sbp"})
    if "dbp" not in bp.columns and "bp_dia" in bp.columns:
        bp = bp.rename(columns={"bp_dia": "dbp"})

    require_columns(bp, ["part_id", "condition", "scenario", "measurement_timestep", "sbp", "dbp"], "BP")

    bp["part_id"] = bp["part_id"].astype(str)
    bp["arm"] = bp["condition"].astype(str)
    bp["scenario"] = bp["scenario"].astype(str)

    bp = bp[bp["arm"].isin(VALID_ARMS)].copy()
    bp = bp[bp["scenario"].isin(VALID_SCENARIOS)].copy()

    bp = ensure_numeric(bp, ["measurement_timestep", "sbp", "dbp"])
    bp["measurement_timestep"] = bp["measurement_timestep"].astype("Int64")

    # Derived metrics
    bp["map"] = (2 * bp["dbp"] + bp["sbp"]) / 3.0
    bp["pp"]  = bp["sbp"] - bp["dbp"]

    # Participant meta
    part = pd.read_csv(PATH_PART)
    part.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in part.columns]
    require_columns(part, ["part_id", "sex"], "Participants")
    part["part_id"] = part["part_id"].astype(str)

    if "fat_pct" in part.columns:
        part["fat_pct"] = pd.to_numeric(part["fat_pct"], errors="coerce")
    else:
        part["fat_pct"] = np.nan

    part["sex"] = part["sex"].astype(str).str.strip().str.upper()
    part["sex_c"] = part["sex"].map({"F": 1, "M": 0, "FEMALE": 1, "MALE": 0}).astype(float)

    bp = bp.merge(part[["part_id", "sex_c", "fat_pct"]], on="part_id", how="left")

    # Categories
    bp["arm"] = pd.Categorical(bp["arm"], categories=ARM_ORDER, ordered=True)
    bp["scenario"] = pd.Categorical(bp["scenario"], categories=SCEN_ORDER, ordered=True)

    return bp, part


# =========================================================
# DRIFT COMPUTATION (measurement_timestep windows)
# =========================================================

def compute_drift_table(bp: pd.DataFrame, value_col: str, metric_label: str,
                       early_steps: list[int], late_steps: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = bp.copy()

    d["is_early"] = d["measurement_timestep"].isin(early_steps)
    d["is_late"]  = d["measurement_timestep"].isin(late_steps)

    gb = ["part_id", "arm", "scenario"]

    early = d[d["is_early"]].groupby(gb).agg(
        n_early=(value_col, "count"),
        early_mean=(value_col, "mean"),
    ).reset_index()

    late = d[d["is_late"]].groupby(gb).agg(
        n_late=(value_col, "count"),
        late_mean=(value_col, "mean"),
    ).reset_index()

    drift = early.merge(late, on=gb, how="outer")
    drift["drift"] = drift["late_mean"] - drift["early_mean"]
    drift["metric"] = metric_label

    # covariates: per participant
    cov = d.groupby("part_id", as_index=False)[["sex_c", "fat_pct"]].first()
    drift = drift.merge(cov, on="part_id", how="left")

    # window completeness exclusions
    excl = drift[
        (drift["n_early"].fillna(0) < MIN_MEAS_PER_WINDOW) |
        (drift["n_late"].fillna(0)  < MIN_MEAS_PER_WINDOW)
    ].copy()
    excl["reason"] = (
        f"{metric_label}_min_meas_fail("
        + "n_early=" + excl["n_early"].fillna(0).astype(int).astype(str)
        + ",n_late=" + excl["n_late"].fillna(0).astype(int).astype(str)
        + ")"
    )

    drift_ok = drift.drop(excl.index).copy()
    drift_ok["arm"] = pd.Categorical(drift_ok["arm"], categories=ARM_ORDER, ordered=True)
    drift_ok["scenario"] = pd.Categorical(drift_ok["scenario"], categories=SCEN_ORDER, ordered=True)

    return drift_ok, excl


# =========================================================
# MODEL FITTING
# =========================================================

def _fit_mixedlm_with_fallback(formula: str, data: pd.DataFrame, reml: bool) -> tuple[object | None, str | None]:
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
    d = drift_df.dropna(subset=["drift", "arm", "scenario", "sex_c", "part_id"]).copy()
    fml = "drift ~ arm * scenario + sex_c"

    res, err = _fit_mixedlm_with_fallback(fml, d, reml=True)
    if res is None:
        raise RuntimeError(f"Final model failed: {err}")
    return res, fml, d


def compute_aic_bic_selection(drift_df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    base = drift_df.dropna(subset=["drift", "arm", "scenario", "part_id"]).copy()
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
                "metric": metric_label,
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
                "metric": metric_label,
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
                "metric": metric_label,
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

    # Add deltas (vs M0)
    for metric in ["aic", "bic"]:
        base_val = out.loc[out["model"] == "M0_arm_x_scen", metric].iloc[0]
        out[f"delta_{metric}_vs_M0"] = out[metric] - base_val if np.isfinite(base_val) else np.nan

    # Mark best within metric
    out["best_aic"] = False
    out["best_bic"] = False
    if out["aic"].notna().any():
        out.loc[out["aic"].idxmin(), "best_aic"] = True
    if out["bic"].notna().any():
        out.loc[out["bic"].idxmin(), "best_bic"] = True

    return out


# =========================================================
# MAIN TEXT TABLE (same logic as SkinTemp)
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

    wide = drift_analysis_df.pivot_table(
        index=["part_id", "arm"], columns="scenario", values="drift", aggfunc="first"
    ).reset_index()

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
# PLOT (same visual language as SkinTemp; jitter off)
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

    # within-arm paired p + dz (simple paired t-test, same as SkinTemp)
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
# RUN PER METRIC
# =========================================================

def run_metric(bp: pd.DataFrame, value_col: str, metric_label: str):
    # Primary drift
    drift_df, excl_df = compute_drift_table(bp, value_col=value_col, metric_label=metric_label,
                                           early_steps=EARLY_STEPS, late_steps=LATE_STEPS)

    drift_df.to_csv(os.path.join(OUTDIR, f"BP_{metric_label}_drift_participant_level.csv"), index=False)
    excl_df.to_csv(os.path.join(OUTDIR, f"BP_{metric_label}_drift_exclusions.csv"), index=False)

    # Sensitivity drift tables (extra outputs; not used for inference unless you choose)
    for name, cfg in SENSITIVITY_DEFS.items():
        dd, ee = compute_drift_table(bp, value_col=value_col, metric_label=f"{metric_label}_{name}",
                                     early_steps=cfg["early"], late_steps=cfg["late"])
        dd.to_csv(os.path.join(OUTDIR, f"BP_{metric_label}_{name}_drift_participant_level.csv"), index=False)
        ee.to_csv(os.path.join(OUTDIR, f"BP_{metric_label}_{name}_drift_exclusions.csv"), index=False)

    # Primary model (REML)
    res, fml, drift_analysis_df = fit_final_model(drift_df)

    with open(os.path.join(OUTDIR, f"BP_{metric_label}_final_model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Final model formula:\n")
        f.write(fml + "\n\n")
        f.write(str(res.summary()))

    main_tbl = build_main_text_table(res, drift_analysis_df)
    main_tbl.to_csv(os.path.join(OUTDIR, f"BP_{metric_label}_Table_MainText.csv"), index=False)
    main_tbl.to_string(open(os.path.join(OUTDIR, f"BP_{metric_label}_Table_MainText.txt"), "w", encoding="utf-8"), index=False)

    # Plot labels
    ylab = f"{metric_label} drift (Late − Early), mmHg"
    title = f"{metric_label} drift by arm and scenario (HS1 vs HS2)"
    plot_paired(drift_analysis_df, OUTDIR, fname_prefix=f"BP_{metric_label}", ylab=ylab, title=title)

    return drift_df, excl_df, res


def run_analysis():
    bp, part = load_and_prepare()

    print("\n=== BP DATASET (after arm/scenario filtering; no QC filtering applied) ===")
    print("Rows:", bp.shape[0])
    print("Participants:", bp["part_id"].nunique())
    print("Arms:", bp["arm"].value_counts(dropna=False).to_dict())
    print("Scenarios:", bp["scenario"].value_counts(dropna=False).to_dict())
    print("Timesteps:", bp["measurement_timestep"].value_counts(dropna=False).sort_index().to_dict())

    model_sel_rows = []

    metrics = [
        ("sbp", "SBP"),
        ("dbp", "DBP"),
        ("map", "MAP"),
        ("pp",  "PP"),
    ]

    for value_col, metric_label in metrics:
        print(f"\n=== RUN METRIC: {metric_label} ===")
        drift_df, excl_df, res = run_metric(bp, value_col=value_col, metric_label=metric_label)

        print("Rows (part×scenario):", drift_df.shape[0])
        print("Participants:", drift_df["part_id"].nunique())
        print("Counts arm×scenario:\n", pd.crosstab(drift_df["arm"], drift_df["scenario"]))

        sel = compute_aic_bic_selection(drift_df, metric_label=metric_label)
        model_sel_rows.append(sel)

    if model_sel_rows:
        model_sel = pd.concat(model_sel_rows, ignore_index=True)
        out_path = os.path.join(OUTDIR, "BP_model_selection_AIC_BIC.csv")
        model_sel.to_csv(out_path, index=False)
        print("\n[MODEL SELECTION] Wrote:", out_path)

    print("\n[DONE] Outputs written to:", OUTDIR)


def main():
    warnings.filterwarnings("ignore")
    run_analysis()


if __name__ == "__main__":
    main()
