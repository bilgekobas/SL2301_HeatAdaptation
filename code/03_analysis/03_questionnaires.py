# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:19:18 2026


Heat Acclimation — Perceptual responses (TSV, TCV, TPV): descriptive-first + paired tests
FINAL (per recommendations) with test results annotated on plots.

Primary inference:
- TSV: Afternoon block PRE (HS1) vs POST (HS2), within-arm paired tests
- TCV: Afternoon block PRE (HS1) vs POST (HS2), within-arm paired tests
- TPV: Afternoon block, binary "prefer cooler" (tpv < 0) PRE vs POST within arm (McNemar)

Descriptive:
- Morning vs Afternoon block-mean plots (TSV/TCV) (no formal inference here)
- TPV full distribution stacked bars (Morning/Afternoon × HS1/HS2)

Annotated on plots:
- TSV/TCV Afternoon plots: within-arm p-values + Cohen’s dz + meanΔ and 95% CI for meanΔ
  (p shown from Wilcoxon when available, else paired t-test)
- TPV McNemar discordant-pairs plot: McNemar p-value per arm + counts

Exports (CSV):
- Perceptual_raw_cleaned.csv
- Perceptual_block_means_participant_level.csv
- TSV_Afternoon_paired_tests.csv
- TCV_Afternoon_paired_tests.csv
- TSV_TCV_Afternoon_paired_tests_combined.csv
- TPV_distribution_proportions.csv
- TPV_coolerPreference_contingency_Afternoon.csv
- TPV_coolerPreference_McNemar_Afternoon.csv

Local paths: keep aligned with your SkinTemp script style.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# Illustrator-friendly vector text
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"



# =========================================================
# REPOSITORY PATHS
# =========================================================
# This file is intended to live in: <repo>/code/03_analysis/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
DATA_METADATA = REPO_ROOT / "data" / "metadata"
OUTPUTS = REPO_ROOT / "outputs" / "model_outputs" / "questionnaires_outputs"

# =========================================================
# CONFIG
# =========================================================

PATH_TC = DATA_PROCESSED / "thermal_comfort.csv"
PATH_PART = DATA_METADATA / "participant_meta_pseud.csv"

OUTDIR = OUTPUTS / "perceptual" / "primary_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

VALID_ARMS = ["FR", "CC"]
ARM_ORDER = ["FR", "CC"]

VALID_SCENARIOS = ["HS1", "HS2"]   # PRE vs POST
SCEN_ORDER = ["HS1", "HS2"]

MORNING_STEPS = [2]
AFTERNOON_STEPS = [5, 8]
TOD_ORDER = ["Morning", "Afternoon"]

# Plot style
PAIR_LINES = True
POINT_SIZE = 46
ERR_LW = 2.2
AXH0_LW = 1.2
LABEL_PARTICIPANTS = True
LABEL_FONTSIZE = 7.5
SAVE_PNG = True
SAVE_PDF = True


# =========================================================
# LABEL→SCALE MAPS (fallback if *_scale missing)
# =========================================================

TSV_MAP = {
    "Cold": -3, "Cool": -2, "Slightly cool": -1,
    "Neither hot nor cold": 0,
    "Slightly warm": 1, "Hot": 2, "Very hot": 3,
}

TCV_MAP = {
    "Comfortable": 0,
    "Slightly uncomfortable": 1,
    "Uncomfortable": 2,
    "Very uncomfortable": 3,
}

TPV_MAP = {
    "Much warmer": 3, "Warmer": 2, "Slightly warmer": 1,
    "No change": 0,
    "Slightly cooler": -1, "Cooler": -2, "Much cooler": -3,
}


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

def parse_arm_from_session_id(session_id: str) -> str | None:
    if session_id is None or (isinstance(session_id, float) and np.isnan(session_id)):
        return None
    s = str(session_id).strip()
    m = re.match(r"^([A-Za-z]+)", s)
    return m.group(1).upper() if m else None

def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)

def mean_diff_ci(diff: np.ndarray, alpha=0.05) -> tuple[float, float]:
    """
    95% CI for mean of paired differences using t distribution.
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n < 2:
        return np.nan, np.nan
    m = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan
    tcrit = float(stats.t.ppf(1 - alpha/2, df=n-1))
    return m - tcrit * se, m + tcrit * se

def paired_tests(pre: np.ndarray, post: np.ndarray):
    """
    Paired t-test + Wilcoxon (if possible), plus dz and CI for mean diff.
    Returns dict with n, mean_pre, mean_post, mean_diff, ci_lo, ci_hi, dz, p_t, p_w, chosen_p, chosen_test.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    m = np.isfinite(pre) & np.isfinite(post)
    pre = pre[m]; post = post[m]
    n = int(pre.size)
    if n < 2:
        return {
            "n": n, "mean_pre": np.nan, "mean_post": np.nan,
            "mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "dz": np.nan, "p_t": np.nan, "p_w": np.nan,
            "chosen_p": np.nan, "chosen_test": "NA"
        }

    diff = post - pre
    dz = cohen_dz(diff)
    mean_pre = float(np.mean(pre))
    mean_post = float(np.mean(post))
    mean_diff = float(np.mean(diff))
    ci_lo, ci_hi = mean_diff_ci(diff)

    # paired t
    t_stat, p_t = stats.ttest_rel(post, pre, nan_policy="omit")
    p_t = float(p_t)

    # wilcoxon (more robust to ordinal nature)
    p_w = np.nan
    try:
        w = stats.wilcoxon(post, pre, zero_method="wilcox", correction=False,
                           alternative="two-sided", mode="auto")
        p_w = float(w.pvalue)
    except Exception:
        p_w = np.nan

    # choose p-value for annotation: Wilcoxon if available, else t-test
    if np.isfinite(p_w):
        chosen_p = p_w
        chosen_test = "Wilcoxon"
    else:
        chosen_p = p_t
        chosen_test = "t-test"

    return {
        "n": n, "mean_pre": mean_pre, "mean_post": mean_post,
        "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "dz": dz, "p_t": p_t, "p_w": p_w,
        "chosen_p": chosen_p, "chosen_test": chosen_test
    }

def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"

def save_fig(fig, prefix: str):
    if SAVE_PNG:
        fig.savefig(os.path.join(OUTDIR, f"{prefix}.png"), dpi=300)
    if SAVE_PDF:
        fig.savefig(os.path.join(OUTDIR, f"{prefix}.pdf"), format="pdf")
    plt.close(fig)


# =========================================================
# LOAD / PREP
# =========================================================

def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PATH_TC)
    df.columns = [c.strip() for c in df.columns]

    req_base = ["part_id", "scenario", "session_id", "measurement_timestep"]
    require_columns(df, req_base, "ThermalComfortFeedback")

    df["part_id"] = df["part_id"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    df["session_id"] = df["session_id"].astype(str)
    df = ensure_numeric(df, ["measurement_timestep"])
    df["measurement_timestep"] = df["measurement_timestep"].astype("Int64")

    # Time-of-day block (avoid dtype promotion by using pandas NA)
    df["tod"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[df["measurement_timestep"].isin(MORNING_STEPS), "tod"] = "Morning"
    df.loc[df["measurement_timestep"].isin(AFTERNOON_STEPS), "tod"] = "Afternoon"
    df = df[df["tod"].notna()].copy()
    df["tod"] = pd.Categorical(df["tod"].astype(str), categories=TOD_ORDER, ordered=True)

    # Participants meta
    part = pd.read_csv(PATH_PART)
    part.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in part.columns]
    require_columns(part, ["part_id", "sex"], "Participants")
    part["part_id"] = part["part_id"].astype(str)

    # Derive arm: prefer participants meta; fallback to session_id prefix
    if "condition" in part.columns:
        arm_map = part[["part_id", "condition"]].rename(columns={"condition": "arm"}).copy()
    elif "arm" in part.columns:
        arm_map = part[["part_id", "arm"]].copy()
    else:
        arm_map = None

    if arm_map is not None:
        arm_map["arm"] = arm_map["arm"].astype(str).str.strip().str.upper()
        df = df.merge(arm_map, on="part_id", how="left")
    else:
        df["arm"] = pd.NA

    miss = df["arm"].isna()
    if miss.any():
        df.loc[miss, "arm"] = df.loc[miss, "session_id"].map(parse_arm_from_session_id)

    df["arm"] = df["arm"].astype(str).str.strip().str.upper()

    # keep FR/CC + HS1/HS2
    df = df[df["arm"].isin(VALID_ARMS)].copy()
    df = df[df["scenario"].isin(VALID_SCENARIOS)].copy()

    # Ensure numeric scale columns exist (fallback mapping from labels)
    def map_if_needed(text_col: str, scale_col: str, mapping: dict):
        if scale_col in df.columns:
            df[scale_col] = pd.to_numeric(df[scale_col], errors="coerce")
        else:
            df[scale_col] = np.nan
        if text_col in df.columns:
            m = df[scale_col].isna() & df[text_col].notna()
            if m.any():
                df.loc[m, scale_col] = df.loc[m, text_col].astype(str).str.strip().map(mapping)

    map_if_needed("tsv", "tsv_scale", TSV_MAP)
    map_if_needed("tcv", "tcv_scale", TCV_MAP)
    map_if_needed("tpv", "tpv_scale", TPV_MAP)

    # Categories
    df["arm"] = pd.Categorical(df["arm"], categories=ARM_ORDER, ordered=True)
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCEN_ORDER, ordered=True)

    # Prevent duplicated col names (defensive)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    return df, part


# =========================================================
# BLOCK MEANS (participant × arm × scenario × tod)
# =========================================================

def build_block_means(df: pd.DataFrame) -> pd.DataFrame:
    gb = ["part_id", "arm", "scenario", "tod"]
    out = (
        df.groupby(gb, as_index=True)
          .agg(
              n=("measurement_timestep", "count"),
              tsv=("tsv_scale", "mean"),
              tcv=("tcv_scale", "mean"),
              tpv=("tpv_scale", "mean"),
              # keep modal label versions for debugging/traceability (optional)
          )
          .reset_index()
    )
    return out


# =========================================================
# PLOTS
# =========================================================

def plot_blockmeans_morning_vs_afternoon(block: pd.DataFrame, y: str, ylabel: str, title: str, prefix: str):
    """
    Descriptive only: Morning and Afternoon panels, PRE vs POST, paired lines and mean±95%CI.
    """
    d = block.dropna(subset=[y, "part_id", "arm", "scenario", "tod"]).copy()
    d["arm"] = pd.Categorical(d["arm"], categories=ARM_ORDER, ordered=True)
    d["scenario"] = pd.Categorical(d["scenario"], categories=SCEN_ORDER, ordered=True)
    d["tod"] = pd.Categorical(d["tod"], categories=TOD_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), sharey=True)

    x_arm = {a: i for i, a in enumerate(ARM_ORDER)}
    x_scen = {"HS1": -0.12, "HS2": 0.12}

    for ax, tod in zip(axes, TOD_ORDER):
        dd = d[d["tod"] == tod].copy()

        # scatter points
        for arm in ARM_ORDER:
            for scen in SCEN_ORDER:
                sub = dd[(dd["arm"] == arm) & (dd["scenario"] == scen)]
                if sub.empty:
                    continue
                x0 = x_arm[arm] + x_scen[scen]
                xs = np.full(len(sub), x0)
                marker = "o" if scen == "HS1" else "s"
                ax.scatter(xs, sub[y].values, s=POINT_SIZE, marker=marker,
                           alpha=0.85, edgecolor="black", linewidth=0.6)

                if LABEL_PARTICIPANTS:
                    for xi, yi, pid in zip(xs, sub[y].values, sub["part_id"].astype(str).values):
                        ax.text(xi, yi, pid, fontsize=LABEL_FONTSIZE, ha="left", va="center")

        # paired lines (within arm)
        if PAIR_LINES:
            wide = dd.pivot_table(index=["part_id", "arm"], columns="scenario", values=y, aggfunc="first").reset_index()
            for _, r in wide.iterrows():
                if pd.isna(r.get("HS1")) or pd.isna(r.get("HS2")):
                    continue
                arm = r["arm"]
                x1 = x_arm[arm] + x_scen["HS1"]
                x2 = x_arm[arm] + x_scen["HS2"]
                ax.plot([x1, x2], [r["HS1"], r["HS2"]], color="black", linewidth=1.0, alpha=0.55)

        # mean ± 95% CI
        summ = dd.groupby(["arm", "scenario"])[y].agg(["count", "mean", "std"]).reset_index()
        summ["se"] = summ["std"] / np.sqrt(summ["count"].clip(lower=1))
        summ["ci_lo"] = summ["mean"] - 1.96 * summ["se"]
        summ["ci_hi"] = summ["mean"] + 1.96 * summ["se"]
        for _, r in summ.iterrows():
            x0 = x_arm[r["arm"]] + x_scen[r["scenario"]]
            marker = "o" if r["scenario"] == "HS1" else "s"
            ax.errorbar([x0], [r["mean"]],
                        yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                        fmt=marker, ecolor="black", color="black",
                        elinewidth=ERR_LW, capsize=4, markersize=6,
                        markerfacecolor="black", markeredgecolor="black", zorder=4)

        ax.axhline(0, color="black", linewidth=AXH0_LW, alpha=0.65)
        ax.set_title(str(tod))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ARM_ORDER)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, prefix)


def plot_afternoon_prepost_with_tests(block: pd.DataFrame, y: str, ylabel: str, title: str,
                                     prefix: str, tests_df_outname: str) -> pd.DataFrame:
    """
    Primary inference plot: Afternoon only, PRE vs POST within arm.
    Annotates per-arm test (Wilcoxon if available else t-test), dz, meanΔ and 95% CI of meanΔ.
    Also writes tests CSV.

    Returns tests table.
    """
    d = block.dropna(subset=[y, "part_id", "arm", "scenario", "tod"]).copy()
    d = d[d["tod"].astype(str).eq("Afternoon")].copy()
    d["arm"] = pd.Categorical(d["arm"], categories=ARM_ORDER, ordered=True)
    d["scenario"] = pd.Categorical(d["scenario"], categories=SCEN_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.9))

    x_arm = {a: i for i, a in enumerate(ARM_ORDER)}
    x_scen = {"HS1": -0.12, "HS2": 0.12}

    # points + labels
    for arm in ARM_ORDER:
        for scen in SCEN_ORDER:
            sub = d[(d["arm"] == arm) & (d["scenario"] == scen)]
            if sub.empty:
                continue
            x0 = x_arm[arm] + x_scen[scen]
            xs = np.full(len(sub), x0)
            marker = "o" if scen == "HS1" else "s"
            ax.scatter(xs, sub[y].values, s=POINT_SIZE, marker=marker,
                       alpha=0.85, edgecolor="black", linewidth=0.6)

            if LABEL_PARTICIPANTS:
                for xi, yi, pid in zip(xs, sub[y].values, sub["part_id"].astype(str).values):
                    ax.text(xi, yi, pid, fontsize=LABEL_FONTSIZE, ha="left", va="center")

    # paired lines
    wide = d.pivot_table(index=["part_id", "arm"], columns="scenario", values=y, aggfunc="first").reset_index()
    if PAIR_LINES:
        for _, r in wide.iterrows():
            if pd.isna(r.get("HS1")) or pd.isna(r.get("HS2")):
                continue
            arm = r["arm"]
            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            ax.plot([x1, x2], [r["HS1"], r["HS2"]], color="black", linewidth=1.0, alpha=0.55)

    # mean ± 95% CI (for each arm×scenario)
    summ = d.groupby(["arm", "scenario"])[y].agg(["count", "mean", "std"]).reset_index()
    summ["se"] = summ["std"] / np.sqrt(summ["count"].clip(lower=1))
    summ["ci_lo"] = summ["mean"] - 1.96 * summ["se"]
    summ["ci_hi"] = summ["mean"] + 1.96 * summ["se"]
    for _, r in summ.iterrows():
        x0 = x_arm[r["arm"]] + x_scen[r["scenario"]]
        marker = "o" if r["scenario"] == "HS1" else "s"
        ax.errorbar([x0], [r["mean"]],
                    yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt=marker, ecolor="black", color="black",
                    elinewidth=ERR_LW, capsize=4, markersize=6,
                    markerfacecolor="black", markeredgecolor="black", zorder=4)

    # tests + annotation (per arm)
    test_rows = []
    y_min, y_max = ax.get_ylim()
    y_span = max(1e-9, (y_max - y_min))
    y_pad = 0.08 * y_span

    for arm in ARM_ORDER:
        w = wide[wide["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any").copy()
        pre = w["HS1"].to_numpy(float) if not w.empty else np.array([])
        post = w["HS2"].to_numpy(float) if not w.empty else np.array([])

        res = paired_tests(pre, post)

        test_rows.append({
            "arm": arm,
            "tod": "Afternoon",
            "n_paired": res["n"],
            "mean_PRE": res["mean_pre"],
            "mean_POST": res["mean_post"],
            "mean_delta_POST_minus_PRE": res["mean_diff"],
            "delta_ci_lo": res["ci_lo"],
            "delta_ci_hi": res["ci_hi"],
            "cohen_dz": res["dz"],
            "p_ttest": res["p_t"],
            "p_wilcoxon": res["p_w"],
            "p_used": res["chosen_p"],
            "p_used_test": res["chosen_test"],
        })

        # bracket + annotation
        if res["n"] >= 2 and np.isfinite(res["chosen_p"]):
            x1 = x_arm[arm] + x_scen["HS1"]
            x2 = x_arm[arm] + x_scen["HS2"]
            # place above max point for this arm
            sub_arm = d[d["arm"] == arm]
            ymax_arm = float(np.nanmax(sub_arm[y].values)) if not sub_arm.empty else y_max
            yb = ymax_arm + y_pad

            ax.plot([x1, x1, x2, x2], [yb - 0.25*y_pad, yb, yb, yb - 0.25*y_pad],
                    color="black", linewidth=1.0, zorder=6)

            txt = (
                f"{res['chosen_test']} {fmt_p(res['chosen_p'])}\n"
                f"dz={res['dz']:.2f}  "
                f"Δ={res['mean_diff']:.2f} [{res['ci_lo']:.2f},{res['ci_hi']:.2f}]  "
                f"n={res['n']}"
            )
            ax.text((x1 + x2) / 2, yb + 0.15*y_pad, txt, ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=AXH0_LW, alpha=0.65)
    ax.set_xticks([x_arm[a] for a in ARM_ORDER])
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, prefix)

    out = pd.DataFrame(test_rows)
    out.to_csv(os.path.join(OUTDIR, tests_df_outname), index=False)
    return out


def plot_tpv_distribution(raw: pd.DataFrame):
    """
    TPV stacked distribution (descriptive).
    """
    d = raw.dropna(subset=["tpv", "arm", "scenario", "tod"]).copy()
    d["tpv"] = d["tpv"].astype(str).str.strip()

    cats = sorted(d["tpv"].unique().tolist(), key=lambda x: TPV_MAP.get(x, 999))

    tab = d.groupby(["arm", "tod", "scenario", "tpv"]).size().reset_index(name="n")
    tab["prop"] = tab.groupby(["arm", "tod", "scenario"])["n"].transform(lambda x: x / x.sum())
    tab.to_csv(os.path.join(OUTDIR, "TPV_distribution_proportions.csv"), index=False)

    combo_order = [(t, s) for t in TOD_ORDER for s in SCEN_ORDER]
    x_labels = [f"{t}\n{s}" for t, s in combo_order]
    x_pos = np.arange(len(combo_order))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharey=True)

    for ax, arm in zip(axes, ARM_ORDER):
        bottom = np.zeros(len(combo_order))
        for cat in cats:
            heights = []
            for (t, s) in combo_order:
                r = tab[(tab["arm"] == arm) & (tab["tod"] == t) & (tab["scenario"] == s) & (tab["tpv"] == cat)]
                heights.append(float(r["prop"].iloc[0]) if not r.empty else 0.0)
            heights = np.array(heights)
            ax.bar(x_pos, heights, bottom=bottom, width=0.65, label=cat,
                   alpha=0.85, edgecolor="black", linewidth=0.4)
            bottom += heights

        ax.set_title(f"TPV distribution — {arm}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Proportion")
    axes[1].legend(title="TPV", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.suptitle("Thermal preference vote (TPV) distribution by arm, time-of-day, and scenario")
    fig.tight_layout(rect=[0, 0, 0.88, 0.93])
    save_fig(fig, "TPV_distribution_stacked")


def tpv_cooler_preference_mcnemar(block: pd.DataFrame):
    """
    Primary inference for TPV: Afternoon-only, binary prefer cooler (tpv < 0), McNemar within arm.
    Also annotates p-values on the discordant-pairs plot.
    """
    aft = block[block["tod"].astype(str).eq("Afternoon")].dropna(subset=["tpv", "arm", "scenario", "part_id"]).copy()
    aft["cooler"] = (aft["tpv"] < 0).astype(int)

    wide = aft.pivot_table(index=["part_id", "arm"], columns="scenario", values="cooler", aggfunc="first").reset_index()

    mcn_rows = []
    cont_rows = []

    for arm in ARM_ORDER:
        w = wide[wide["arm"] == arm].dropna(subset=["HS1", "HS2"], how="any").copy()
        n_paired = int(w.shape[0])

        a11 = int(((w["HS1"] == 1) & (w["HS2"] == 1)).sum()) if n_paired else 0
        b10 = int(((w["HS1"] == 1) & (w["HS2"] == 0)).sum()) if n_paired else 0
        c01 = int(((w["HS1"] == 0) & (w["HS2"] == 1)).sum()) if n_paired else 0
        d00 = int(((w["HS1"] == 0) & (w["HS2"] == 0)).sum()) if n_paired else 0

        cont_rows.append({"arm": arm, "1→1": a11, "1→0": b10, "0→1": c01, "0→0": d00, "n_paired": n_paired})

        if n_paired == 0:
            mcn_rows.append({
                "arm": arm, "n_paired": 0, "b_pre1_post0": np.nan, "c_pre0_post1": np.nan,
                "mcnemar_stat": np.nan, "p_value": np.nan, "exact": np.nan, "note": "no_paired_data"
            })
            continue

        exact = (b10 + c01) < 25
        try:
            res = mcnemar([[a11, b10], [c01, d00]], exact=exact, correction=not exact)
            stat = float(res.statistic) if hasattr(res, "statistic") else np.nan
            pval = float(res.pvalue) if hasattr(res, "pvalue") else np.nan
        except Exception:
            stat, pval = np.nan, np.nan

        mcn_rows.append({
            "arm": arm, "n_paired": n_paired,
            "b_pre1_post0": b10, "c_pre0_post1": c01,
            "mcnemar_stat": stat, "p_value": pval,
            "exact": exact, "note": ""
        })

    mcn = pd.DataFrame(mcn_rows)
    cont = pd.DataFrame(cont_rows)

    mcn.to_csv(os.path.join(OUTDIR, "TPV_coolerPreference_McNemar_Afternoon.csv"), index=False)
    cont.to_csv(os.path.join(OUTDIR, "TPV_coolerPreference_contingency_Afternoon.csv"), index=False)

    # Discordant-pairs plot with p-values
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    x = np.arange(len(ARM_ORDER))
    wbar = 0.35

    ax.bar(x - wbar/2, cont["1→0"], width=wbar, label="1→0 (cooler→not cooler)",
           alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.bar(x + wbar/2, cont["0→1"], width=wbar, label="0→1 (not cooler→cooler)",
           alpha=0.85, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(ARM_ORDER)
    ax.set_ylabel("Discordant pair count")
    ax.set_title("TPV cooler-preference changes (Afternoon): discordant pairs (McNemar)")
    ax.legend(frameon=False)

    # annotate n and p
    ymax = float(max(cont["1→0"].max(), cont["0→1"].max(), 0))
    ytxt = ymax + 0.15 if ymax < 6 else ymax + 0.25

    for i, arm in enumerate(ARM_ORDER):
        n = int(cont.loc[cont["arm"] == arm, "n_paired"].iloc[0])
        p = mcn.loc[mcn["arm"] == arm, "p_value"].iloc[0]
        ax.text(i, ytxt, f"n={n}\n{fmt_p(p)}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    save_fig(fig, "TPV_coolerPreference_discordant_counts")


# =========================================================
# RUN
# =========================================================

def run_analysis():
    raw, part = load_and_prepare()

    # Export cleaned raw
    raw.to_csv(os.path.join(OUTDIR, "Perceptual_raw_cleaned.csv"), index=False)

    # Block means
    block = build_block_means(raw)
    block.to_csv(os.path.join(OUTDIR, "Perceptual_block_means_participant_level.csv"), index=False)

    print("\n=== PERCEPTUAL DATASET ===")
    print("Raw rows:", raw.shape[0])
    print("Participants:", raw["part_id"].nunique())
    print("Arms:", raw["arm"].value_counts(dropna=False).to_dict())
    print("Scenarios:", raw["scenario"].value_counts(dropna=False).to_dict())
    print("Timesteps:", raw["measurement_timestep"].value_counts(dropna=False).sort_index().to_dict())

    # Descriptive: Morning vs Afternoon block means
    plot_blockmeans_morning_vs_afternoon(
        block, y="tsv",
        ylabel="TSV (−3..+3)",
        title="Thermal sensation (TSV) block means by arm and scenario",
        prefix="TSV_blockmeans_paired"
    )
    plot_blockmeans_morning_vs_afternoon(
        block, y="tcv",
        ylabel="TCV (0..4)",
        title="Thermal comfort (TCV) block means by arm and scenario",
        prefix="TCV_blockmeans_paired"
    )

    # Primary inference: Afternoon PRE vs POST with tests annotated
    tsv_tests = plot_afternoon_prepost_with_tests(
        block, y="tsv",
        ylabel="TSV (−3..+3)",
        title="TSV (Afternoon block): PRE vs POST by arm",
        prefix="TSV_Afternoon_paired_with_tests",
        tests_df_outname="TSV_Afternoon_paired_tests.csv"
    )
    tcv_tests = plot_afternoon_prepost_with_tests(
        block, y="tcv",
        ylabel="TCV (0..4)",
        title="TCV (Afternoon block): PRE vs POST by arm",
        prefix="TCV_Afternoon_paired_with_tests",
        tests_df_outname="TCV_Afternoon_paired_tests.csv"
    )

    comb = pd.concat(
        [tsv_tests.assign(outcome="TSV"), tcv_tests.assign(outcome="TCV")],
        ignore_index=True
    )
    comb.to_csv(os.path.join(OUTDIR, "TSV_TCV_Afternoon_paired_tests_combined.csv"), index=False)

    # TPV descriptive + primary inference (McNemar)
    plot_tpv_distribution(raw)
    tpv_cooler_preference_mcnemar(block)

    print("\n[DONE] Outputs written to:", OUTDIR)


def main():
    warnings.filterwarnings("ignore")
    run_analysis()


if __name__ == "__main__":
    main()
