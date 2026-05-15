# -*- coding: utf-8 -*-
"""
00_HRV_DetectMovement_public.py

Create activity summaries from Shimmer EXG files and generate HRV sedentary/QC flags.

Outputs:
    data/processed/activity.csv
    data/processed/activity_5min.csv

Raw files are expected under:
    data/raw/session_data/**/*_Calibrated_SD.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_SESSION_ROOT = DATA_DIR / "raw" / "session_data"
PROCESSED_DIR = DATA_DIR / "processed"

TZ = "Europe/Berlin"

SUFFIXES = [
    "Accel_LN_X_CAL", "Accel_LN_Y_CAL", "Accel_LN_Z_CAL",
    "Accel_WR_X_CAL", "Accel_WR_Y_CAL", "Accel_WR_Z_CAL",
    "Activity_Accel_Magnitude_CAL",
    "Activity_Intensity_Raw_CAL",
    "Activity_Intensity_Scaled_CAL",
    "Activity_Percentage_Active_CAL",
    "Activity_Percentage_Sedentary_CAL",
    "Activity_Step_Count_CAL",
]

QC_SEDENTARY_MIN = 0.9
QC_ACCEL_STD_MAX = 0.5
TIMESTAMP_SUFFIX = "_Timestamp_Unix_CAL"
CAND_SEPS = [",", ";", "\t", "|"]


def find_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*_Calibrated_SD.csv"))


def detect_sep_and_skiprows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        first = f.readline()

    norm = first.lstrip("\ufeff").strip().strip('"').strip("'")
    m = re.match(r"(?i)^\s*sep\s*=\s*(.+?)\s*$", norm)
    if m:
        token = m.group(1).replace("\\t", "\t")
        if token.lower() == "tab":
            token = "\t"
        sep = token if token else ","
        return sep, [0], [0, 2], True

    best_sep, best_n = ",", 1
    for s in CAND_SEPS:
        n = len(first.split(s))
        if n > best_n:
            best_sep, best_n = s, n
    return best_sep, [], [1], False


def load_csv_skip_units(path: Path) -> pd.DataFrame:
    sep, _, skiprows_data, _ = detect_sep_and_skiprows(str(path))
    df = pd.read_csv(path, sep=sep, header=0, skiprows=skiprows_data, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_ecg_id_from_cols(columns) -> str | None:
    for c in columns:
        m = re.search(r"(SL\d{3})", c)
        if m:
            return m.group(1)
    return None


def detect_ecg_id_from_path(path: Path) -> str | None:
    for part in path.parts[::-1]:
        m = re.search(r"(SL\d{3})", part)
        if m:
            return m.group(1)
    return None


def detect_prefix(columns, ecg_id):
    if ecg_id is None:
        return None
    for c in columns:
        m = re.match(r"([A-Za-z0-9]+_" + re.escape(ecg_id) + r")_", c)
        if m:
            return m.group(1)
    return None


def choose_activity_columns(df: pd.DataFrame, ecg_prefix: str | None) -> list[str]:
    chosen = []
    for suffix in SUFFIXES:
        if ecg_prefix and f"{ecg_prefix}_{suffix}" in df.columns:
            chosen.append(f"{ecg_prefix}_{suffix}")
        else:
            matches = [c for c in df.columns if c.endswith("_" + suffix)]
            if matches:
                chosen.append(matches[0])
    return [c for c in chosen if c in df.columns]


def _parse_timestamp_to_ms(ts_series: pd.Series) -> pd.Series:
    ts_num = pd.to_numeric(ts_series, errors="coerce")
    if ts_num.notna().mean() > 0.9:
        med = ts_num.median()
        if med > 1e12:
            return ts_num.astype("float64")
        if med > 1e9:
            return (ts_num * 1000.0).astype("float64")
        return (ts_num * 1000.0).astype("float64")

    ts_dt = pd.to_datetime(ts_series, errors="coerce", utc=True)
    ts_ns = ts_dt.view("int64")
    ms = np.where(ts_dt.notna(), ts_ns // 10**6, np.nan).astype("float64")
    return pd.Series(ms, index=ts_series.index, dtype="float64")


def clean_columns(cols: list[str]) -> list[str]:
    newcols = []
    for c in cols:
        if re.search(r"^[A-Za-z0-9]+_[A-Za-z0-9]+_", c):
            newcols.append(c.split("_", 2)[-1])
        else:
            newcols.append(c)
    return newcols


def summarize_one_file(csv_path: Path):
    df_raw = load_csv_skip_units(csv_path)
    if df_raw.empty:
        print(f"[SKIP] {csv_path} (empty)")
        return None, None

    cols = df_raw.columns.tolist()
    ts_candidates = [c for c in cols if c.endswith(TIMESTAMP_SUFFIX)]
    if not ts_candidates:
        ts_candidates = [c for c in cols if re.search(r"Timestamp.*Unix.*_CAL$", c)]
    if not ts_candidates:
        ts_candidates = [c for c in cols if re.search(r"Timestamp", c, flags=re.I)]
    if not ts_candidates:
        ts_candidates = [c for c in cols if re.search(r"(Time|Datetime|DateTime)", c, flags=re.I)]
    if not ts_candidates:
        print(f"[WARN] No timestamp column in {csv_path.name}")
        return None, None

    ts_col = ts_candidates[0]
    ecg_id = detect_ecg_id_from_cols(df_raw.columns) or detect_ecg_id_from_path(csv_path) or "SLUNK"
    prefix = detect_prefix(df_raw.columns, ecg_id)

    activity_cols = choose_activity_columns(df_raw, prefix)
    if not activity_cols:
        print(f"[WARN] No activity columns found in {csv_path.name}")
        return None, None

    df = df_raw[[ts_col] + activity_cols].copy()
    ts_ms = _parse_timestamp_to_ms(df[ts_col])
    ok = ts_ms.notna()
    if not ok.any():
        print(f"[WARN] Could not parse timestamp in {csv_path.name}")
        return None, None

    dt_utc = pd.to_datetime(ts_ms[ok], unit="ms", utc=True)
    df = df.loc[ok].copy()
    df["timestamp"] = dt_utc.dt.tz_convert(TZ)

    for c in activity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == -1, c] = np.nan

    activity_cols = [c for c in activity_cols if not df[c].isna().all()]
    if not activity_cols:
        return None, None

    col_step = next((c for c in activity_cols if c.endswith("Activity_Step_Count_CAL")), None)
    col_sed = next((c for c in activity_cols if c.endswith("Activity_Percentage_Sedentary_CAL")), None)
    col_accm = next((c for c in activity_cols if c.endswith("Activity_Accel_Magnitude_CAL")), None)

    df["minute_start"] = df["timestamp"].dt.floor("min")

    if col_accm:
        accel_std_1m = df.groupby("minute_start")[col_accm].std(ddof=0).rename("AccelMag_std")
    else:
        accel_std_1m = pd.Series(dtype=float)

    agg_map_1m = {c: "mean" for c in activity_cols}
    if col_step:
        agg_map_1m[col_step] = "sum"

    one_min = df.groupby("minute_start", as_index=False).agg(agg_map_1m)
    one_min.insert(1, "ecg_id", ecg_id)

    if not accel_std_1m.empty:
        one_min = one_min.merge(accel_std_1m.reset_index(), on="minute_start", how="left")
    else:
        one_min["AccelMag_std"] = np.nan

    step_ok = (one_min[col_step] <= 5) if col_step else True
    one_min["valid_for_HRV"] = step_ok

    om = one_min.sort_values("minute_start").set_index("minute_start")
    step_cols = [col_step] if col_step else []
    cont_cols = [c for c in activity_cols if c not in step_cols]

    roll_mean = om[cont_cols].rolling(window=5, min_periods=5).mean() if cont_cols else pd.DataFrame(index=om.index)
    roll_sum = om[step_cols].rolling(window=5, min_periods=5).sum() if step_cols else pd.DataFrame(index=om.index)
    roll_acc_std = om[["AccelMag_std"]].rolling(window=5, min_periods=5).mean()

    five_min = pd.concat([roll_mean, roll_sum, roll_acc_std], axis=1).dropna(how="all")
    if not five_min.empty:
        five_min = five_min.reset_index()
        five_min.insert(1, "ecg_id", ecg_id)
        five_min["window_end"] = five_min["minute_start"] + pd.Timedelta(minutes=1)
        five_min["window_start"] = five_min["minute_start"] - pd.Timedelta(minutes=4)

        step5 = five_min[col_step] if col_step else pd.Series(0, index=five_min.index)
        sed5 = five_min[col_sed] if col_sed else pd.Series(100, index=five_min.index)
        acc5 = five_min["AccelMag_std"].fillna(0)
        five_min["valid_for_HRV"] = (step5 == 0) & (sed5 >= QC_SEDENTARY_MIN) & (acc5 <= QC_ACCEL_STD_MAX)

        cols = ["window_start", "window_end", "ecg_id", "valid_for_HRV"] + [
            c for c in five_min.columns if c not in {"window_start", "window_end", "ecg_id", "valid_for_HRV"}
        ]
        five_min = five_min[cols]
    else:
        five_min = None

    base = csv_path.name.replace("_Calibrated_SD.csv", "")
    out1 = csv_path.with_name(f"{base}_Activity_1min.csv")
    one_min.to_csv(out1, index=False)

    out5 = None
    if five_min is not None:
        out5 = csv_path.with_name(f"{base}_Activity_5min.csv")
        five_min.to_csv(out5, index=False)

    return out1, out5


def concat_activity(root: Path, suffix: str, outdir: Path, outname: str):
    files = sorted(root.rglob(f"*Activity_{suffix}.csv"))
    if not files:
        print(f"No {suffix} files found under {root}")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = clean_columns(df.columns)
            df["source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")

    if not dfs:
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / outname
    all_df.to_csv(outpath, index=False)
    print(f"[OK] Wrote {outpath} with {len(all_df)} rows from {len(files)} files.")
    return all_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Shimmer activity summaries.")
    parser.add_argument("--raw-root", type=Path, default=RAW_SESSION_ROOT, help="Raw session-data root.")
    args = parser.parse_args()

    files = find_files(args.raw_root)
    if not files:
        raise FileNotFoundError(f"No *_Calibrated_SD.csv files found under {args.raw_root}")

    print(f"Found {len(files)} file(s).")
    for f in files:
        try:
            out1, out5 = summarize_one_file(f)
            if out1 is None and out5 is None:
                print(f"[SKIP] {f}")
            else:
                outs = [p.name for p in (out1, out5) if p]
                print(f"[OK] {f} -> {', '.join(outs)}")
        except Exception as e:
            print(f"[ERR] {f}: {e}")

    concat_activity(args.raw_root, "1min", PROCESSED_DIR, "activity.csv")
    concat_activity(args.raw_root, "5min", PROCESSED_DIR, "activity_5min.csv")


if __name__ == "__main__":
    main()
