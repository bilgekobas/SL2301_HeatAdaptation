# -*- coding: utf-8 -*-
"""
00_Climate_toppmv_plots_public.py

Calculate MRT, operative temperature, PMV, exposure temperature, climate summaries,
and FR-anchored heat/cool dose metrics.

Expected input:
    data/processed/01_ClimateData_merged.csv

Outputs:
    data/processed/01_ClimateData_merged_withtop.csv
    outputs/climate/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "climate"

INPUT_FILE = PROCESSED_DIR / "01_ClimateData_merged.csv"
OUTPUT_FILE = PROCESSED_DIR / "01_ClimateData_merged_withtop.csv"
SUMMARY_CSV = OUTPUTS_DIR / "climate_summaries_mean_sd_by_condition.csv"

MET = 1.1
CLO = 0.5
AIR_SPEED = 0.1

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def calc_mrt(tg, tdb, v):
    """ISO 7726 approximation for mean radiant temperature from globe temperature."""
    tgK = tg + 273.15
    tdbK = tdb + 273.15
    mrtK = (tgK**4 + (1.1e8 * (v**0.6)) * (tg - tdb) / tdbK) ** 0.25
    return mrtK - 273.15


def calc_pmv(tdb, tr, vr, rh, met, clo, wme=0):
    """Fanger PMV. tdb,tr °C; vr m/s; rh %; met met; clo clo."""
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (tdb + 235))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w
    fcl = (1 + 1.29 * icl) if icl <= 0.078 else (1.05 + 0.645 * icl)
    taa = tdb + 273
    tra = tr + 273
    tcla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * ((tra / 100) ** 4)

    xn = tcla / 100
    xf = tcla / 50
    eps = 0.00015
    n = 0

    while abs(xn - xf) > eps and n < 150:
        xf = (xf + xn) / 2
        hcf = 12.1 * np.sqrt(vr)
        hc = hcf if (2.38 * abs(100 * xf - taa)) ** 0.25 < hcf else (2.38 * abs(100 * xf - taa)) ** 0.25
        xn = (p5 + p4 * hc - p2 * (xf ** 4)) / (100 + p3 * hc)
        n += 1

    hl1 = 3.05e-3 * (5733 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0
    hl3 = 1.7e-5 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - tdb)
    hl5 = 3.96 * fcl * (xn**4 - (tra / 100) ** 4)
    hl6 = fcl * ((12.1 * np.sqrt(vr)) if True else 0) * (100 * xn - 273 - tdb)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    return ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)


def norm_cond(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip()
              .str.replace("^Fr$", "FR", regex=True)
              .str.replace("^Hs$", "HS", regex=True)
              .str.replace("^Mix$", "DC", regex=True)
              .str.replace("^AC$", "CC", regex=True)
              .str.replace("^Cc$", "CC", regex=True))


def add_derived_climate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if "tr" not in df.columns:
        if not {"tg", "tdb"}.issubset(df.columns):
            raise KeyError("Need 'tg' and 'tdb' to compute 'tr'.")
        df["tr"] = df.apply(lambda r: calc_mrt(r["tg"], r["tdb"], AIR_SPEED), axis=1)

    if "top" not in df.columns:
        df["top"] = (df["tdb"] * np.sqrt(10 * AIR_SPEED) + df["tr"]) / (1 + np.sqrt(10 * AIR_SPEED))

    if "pmv" not in df.columns:
        if "rh" not in df.columns:
            raise KeyError("Need 'rh' to compute PMV.")
        df["pmv"] = df.apply(lambda r: calc_pmv(r["tdb"], r["tr"], AIR_SPEED, r["rh"], MET, CLO), axis=1)

    if "condition" not in df.columns:
        raise KeyError("Need 'condition' column to compute texp.")
    df["condition"] = norm_cond(df["condition"])

    required = {"top", "tout_2m", "tout_30m"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required for texp: {missing}")

    t = df["datetime"].dt.time
    lunch = (t >= dt.time(12, 30)) & (t < dt.time(13, 30))

    df["texp"] = df["top"].astype(float)
    df.loc[(df["condition"] == "HS") & lunch, "texp"] = df.loc[(df["condition"] == "HS") & lunch, "tout_30m"].astype(float)
    df.loc[(df["condition"] == "FR") & lunch, "texp"] = df.loc[(df["condition"] == "FR") & lunch, "tout_2m"].astype(float)
    df.loc[(df["condition"] == "DC") & lunch, "texp"] = df.loc[(df["condition"] == "DC") & lunch, "tout_2m"].astype(float)

    return df


def write_summaries(df: pd.DataFrame, out_csv: Path = SUMMARY_CSV) -> pd.DataFrame:
    summary_vars = [v for v in ["texp", "top", "tdb", "tr", "rh", "co2", "pmv"] if v in df.columns]
    rows = []

    if "condition" in df.columns:
        cond_tab = df.groupby("condition", dropna=False)[summary_vars].agg(["mean", "std"])
        cond_tab.columns = [f"{v}_{s}" for v, s in cond_tab.columns]
        cond_tab.insert(0, "group", cond_tab.index.astype(str))
        rows.append(cond_tab.reset_index(drop=True))

    if "condition" in df.columns and "scenario_short" in df.columns:
        hs_df = df.loc[df["condition"].eq("HS")].copy()
        if not hs_df.empty:
            sub = hs_df.groupby("scenario_short", dropna=False)[summary_vars].agg(["mean", "std"])
            sub.columns = [f"{v}_{s}" for v, s in sub.columns]
            sub.insert(0, "group", sub.index.astype(str))
            rows.append(sub.reset_index(drop=True))

    if not rows:
        raise RuntimeError("No summaries produced; check columns.")

    out = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")
    return out


def compute_fr_anchor_doses(df: pd.DataFrame, target_var: str = "texp", out_dir: Path = OUTPUTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    need = {"datetime", "condition", target_var}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.dropna(subset=["datetime", "condition", target_var]).copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["condition"] = norm_cond(d["condition"])

    d = d.sort_values(["condition", "datetime"]).reset_index(drop=True)
    dtmin = d.groupby("condition")["datetime"].diff().dt.total_seconds().fillna(60.0) / 60.0
    d["minutes"] = dtmin.clip(0.5, 5.0)

    fr_vals = d.loc[d["condition"] == "FR", target_var]
    if fr_vals.empty:
        raise ValueError("No FR rows found; cannot compute FR anchors.")

    theta_fr = float(fr_vals.mean())
    q1 = float(fr_vals.quantile(0.25))
    q3 = float(fr_vals.quantile(0.75))

    d["delta_vs_FR_mean"] = d[target_var] - theta_fr
    d["HDM_FRmean_row"] = d["delta_vs_FR_mean"].clip(lower=0) * d["minutes"]
    d["CDM_FRmean_row"] = (-d["delta_vs_FR_mean"]).clip(lower=0) * d["minutes"]

    d["HDM_FRIQR_row"] = (d[target_var] - q3).clip(lower=0) * d["minutes"]
    d["CDM_FRIQR_row"] = (q1 - d[target_var]).clip(lower=0) * d["minutes"]

    def summarize(h_col, c_col):
        out = d.groupby("condition", as_index=False).agg(
            HDM=(h_col, "sum"),
            CDM=(c_col, "sum"),
            minutes_total=("minutes", "sum"),
        )
        out["HDM_degh"] = out["HDM"] / 60.0
        out["CDM_degh"] = out["CDM"] / 60.0
        out["HDM_per_hr"] = out["HDM"] / (out["minutes_total"] / 60.0)
        out["CDM_per_hr"] = out["CDM"] / (out["minutes_total"] / 60.0)
        return out

    mean_summary = summarize("HDM_FRmean_row", "CDM_FRmean_row")
    iqr_summary = summarize("HDM_FRIQR_row", "CDM_FRIQR_row")

    d.to_csv(out_dir / f"rowlevel_with_doses_{target_var}.csv", index=False)
    mean_summary.to_csv(out_dir / f"FRmean_anchor_summary_{target_var}.csv", index=False)
    iqr_summary.to_csv(out_dir / f"FRIQR_anchor_summary_{target_var}.csv", index=False)

    print(f"[OK] FR mean anchor = {theta_fr:.2f} °C")
    print(f"[OK] FR IQR anchor = {q1:.2f}–{q3:.2f} °C")
    print(f"[OK] Dose outputs written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute climate exposure variables and dose summaries.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Climate input file not found: {args.input}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = add_derived_climate(df)
    df.to_csv(args.output, index=False)
    print(f"[OK] Wrote {args.output}")

    write_summaries(df, SUMMARY_CSV)
    compute_fr_anchor_doses(df, target_var="texp", out_dir=OUTPUTS_DIR)


if __name__ == "__main__":
    main()
