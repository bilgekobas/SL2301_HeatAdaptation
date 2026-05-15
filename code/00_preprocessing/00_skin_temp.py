# -*- coding: utf-8 -*-
"""
00_SkinTemp_public.py

Merge iButton skin-temperature files, clip by session metadata, merge neck+ankle,
add QC flags, and save:
    data/processed/skin_temp.csv

The raw iButton files are expected under:
    data/raw/session_data/**/SkinTemp/*_formatted.csv

If raw files are not included in the public repository, this script documents the
preprocessing workflow but cannot be executed until raw files are supplied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import argparse
import re
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_SESSION_ROOT = DATA_DIR / "raw" / "session_data"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
OUT_DIR = PROCESSED_DIR / "outputs" / "model_outputs" / "skin_temp"

SESSION_META_CSV = METADATA_DIR / "session_meta.csv"
OUTPUT_CSV = OUT_DIR / "skin_temp.csv"

META_PART_ID_COL_CANDIDATES = ["part_id", "participant", "id"]
META_DATE_COL_CANDIDATES = ["date", "session_date"]
META_SESSION_COL_CANDIDATES = ["session_id", "session", "condition", "hs_label"]
META_START_KEYS = ["start", "session_start", "hs_start"]
META_END_KEYS = ["end", "session_end", "hs_end"]

ASSUME_YEAR_PREFIX = "20"
FORCE_JUNE = True
DAYFIRST_DEFAULT = True
QC_DROP_PER_MIN_THRESHOLD = 0.8
VERBOSE = True


def _print(msg: str) -> None:
    if VERBOSE:
        print(msg)


def find_first_matching_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def find_col_by_keywords(cols: List[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        cl = c.lower()
        if any(k.lower() in cl for k in keywords):
            return c
    return None


def sanitize_two_digit_year(s: str) -> str:
    if not ASSUME_YEAR_PREFIX:
        return s
    m = re.search(r"(^|\D)(\d{1,2})[./\-](\d{1,2})[./\-](\d{2})($|\D)", s.strip())
    if m:
        yy = m.group(4)
        return re.sub(r"(\d{1,2})([./\-])(\d{1,2})([./\-])(\d{2})",
                      rf"\1\2\3\4{ASSUME_YEAR_PREFIX}{yy}", s)
    return s


def parse_date_assuming_june(val) -> pd.Timestamp:
    if isinstance(val, pd.Timestamp):
        dt = val.normalize()
        return dt if (not FORCE_JUNE or dt.month == 6) else dt.replace(month=6)

    s = str(val).strip()
    if not s or s.lower() in ["nan", "nat"]:
        return pd.NaT

    s2 = sanitize_two_digit_year(s)
    a = pd.to_datetime(s2, errors="coerce", dayfirst=DAYFIRST_DEFAULT)
    b = pd.to_datetime(s2, errors="coerce", dayfirst=not DAYFIRST_DEFAULT)

    if FORCE_JUNE:
        if pd.notna(a) and getattr(a, "month", None) == 6:
            return a.normalize()
        if pd.notna(b) and getattr(b, "month", None) == 6:
            return b.normalize()
        if pd.notna(a):
            return a.normalize().replace(month=6)
        if pd.notna(b):
            return b.normalize().replace(month=6)
        return pd.NaT

    return a.normalize() if pd.notna(a) else (b.normalize() if pd.notna(b) else pd.NaT)


def infer_session_id_from_path(path: Path) -> str:
    parts = path.parts
    try:
        idx = len(parts) - 1 - parts[::-1].index("SkinTemp")
    except ValueError:
        return path.parent.name
    sess = parts[idx - 1] if idx - 1 >= 0 else path.parent.name
    return re.sub(r"[,\s]+", "_", sess)


def load_sessions_meta(path: Path = SESSION_META_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Session metadata not found: {path}")
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df


def build_datetime_from_date_time(date_col: pd.Series, time_col: Optional[pd.Series]) -> pd.Series:
    dts = date_col.apply(parse_date_assuming_june)
    if time_col is None:
        return dts
    parsed_t = pd.to_datetime(time_col.astype(str), errors="coerce")
    out = dts.copy()
    has_time = parsed_t.notna()
    out.loc[has_time] = pd.to_datetime(
        dts.loc[has_time].dt.date.astype(str) + " " + parsed_t.loc[has_time].dt.time.astype(str),
        errors="coerce",
    )
    return out


def standardize_skin_file(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath)

    for col in ["Time_raw", "iButtonID"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    ren = {}
    if "Date" in df.columns:
        ren["Date"] = "date"
    if "Time" in df.columns:
        ren["Time"] = "time"
    if "Datetime" in df.columns:
        ren["Datetime"] = "datetime"
    if "Participant" in df.columns:
        ren["Participant"] = "part_id"
    df = df.rename(columns=ren)

    if "part_id" not in df.columns:
        m = re.search(r"(P\d+)", fpath.name, flags=re.IGNORECASE)
        df["part_id"] = m.group(1).upper() if m else "UNKNOWN"

    location_val = ""
    if "Location" in df.columns:
        loc = df["Location"].dropna().astype(str).str.lower()
        location_val = loc.mode().iloc[0] if len(loc) else ""

    fname = fpath.name.lower()
    if "neck" in location_val or "neck" in fname:
        sensor_col = "skin_temp_neck"
    elif "ankle" in location_val or "ankle" in fname:
        sensor_col = "skin_temp_ankle"
    else:
        sensor_col = "skin_temp_unknown"

    temp_source = None
    for c in df.columns:
        if c.lower() in ["skintemp", "skin_temp", "temperature", "temp"]:
            temp_source = c
            break
    if temp_source is None:
        raise ValueError(f"{fpath.name}: no temperature column found.")
    if temp_source != sensor_col:
        df = df.rename(columns={temp_source: sensor_col})

    if "date" in df.columns:
        df["date"] = df["date"].apply(parse_date_assuming_june).dt.date
    else:
        raise ValueError(f"{fpath.name}: missing Date/date column.")

    time_series = df["time"] if "time" in df.columns else None
    df["datetime"] = build_datetime_from_date_time(df["date"].astype(str), time_series)
    df["session_id"] = infer_session_id_from_path(fpath)

    if "Location" in df.columns:
        df = df.drop(columns=["Location"])

    keep = ["datetime", "date", "time", "part_id", "session_id", sensor_col]
    df = df[keep].copy()
    return df.sort_values(["part_id", "date", "session_id", "datetime"]).drop_duplicates()


def clip_to_session_times(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = meta.columns.tolist()
    part_col = find_first_matching_col(cols, META_PART_ID_COL_CANDIDATES) or "part_id"
    date_col = find_first_matching_col(cols, META_DATE_COL_CANDIDATES) or "date"
    sess_col = find_first_matching_col(cols, META_SESSION_COL_CANDIDATES)
    start_col = find_col_by_keywords(cols, META_START_KEYS)
    end_col = find_col_by_keywords(cols, META_END_KEYS)

    if not start_col or not end_col:
        _print("[WARN] Start/End columns not found in session_meta.csv. Skipping clipping.")
        return df

    meta = meta.copy()
    meta["_date_norm"] = meta[date_col].apply(parse_date_assuming_june).dt.date

    def pick_meta(pid, d, sid) -> Optional[pd.Series]:
        sub = meta[meta[part_col].astype(str).str.upper() == str(pid).upper()]
        sub = sub[sub["_date_norm"] == d]
        if sess_col and sid:
            sid_l = str(sid).lower()
            hit = sub[sub[sess_col].astype(str).str.lower() == sid_l]
            if not hit.empty:
                return hit.iloc[0]
            hit = sub[sub[sess_col].astype(str).str.lower().str.contains(re.escape(sid_l), na=False)]
            if not hit.empty:
                return hit.iloc[0]
        if not sub.empty:
            return sub.iloc[0]
        return None

    clipped = []
    missing = 0
    for (pid, d, sid), g in df.groupby(["part_id", "date", "session_id"], sort=False):
        row = pick_meta(pid, d, sid)
        if row is None:
            missing += 1
            clipped.append(g)
            continue

        base_date = parse_date_assuming_june(row[date_col])
        start_dt = pd.to_datetime(str(base_date.date()) + " " + str(row[start_col]), errors="coerce")
        end_dt = pd.to_datetime(str(base_date.date()) + " " + str(row[end_col]), errors="coerce")

        if pd.isna(start_dt) or pd.isna(end_dt):
            missing += 1
            clipped.append(g)
            continue

        clipped.append(g[(g["datetime"] >= start_dt) & (g["datetime"] <= end_dt)])

    if missing:
        _print(f"[WARN] {missing} group(s) could not be clipped using metadata.")

    return pd.concat(clipped, ignore_index=True)


def merge_neck_ankle(df: pd.DataFrame) -> pd.DataFrame:
    key = ["part_id", "date", "session_id", "datetime", "time"]
    neck_cols = key + [c for c in df.columns if c == "skin_temp_neck"]
    ankle_cols = key + [c for c in df.columns if c == "skin_temp_ankle"]

    both = pd.concat([df[neck_cols], df[ankle_cols]], ignore_index=True)
    return (both.sort_values(key)
                .groupby(key, as_index=False)
                .agg(skin_temp_neck=("skin_temp_neck", "first"),
                     skin_temp_ankle=("skin_temp_ankle", "first")))


def add_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["part_id", "date", "session_id", "datetime"])
    df["neck_diff"] = np.nan
    df["ankle_diff"] = np.nan
    df["qc_neck_sudden_drop"] = False
    df["qc_ankle_sudden_drop"] = False

    for (_, _, _), gidx in df.groupby(["part_id", "date", "session_id"]).groups.items():
        idx = list(gidx)

        neck = df.loc[idx, "skin_temp_neck"]
        if neck.notna().any():
            dneck = neck.diff()
            df.loc[idx, "neck_diff"] = dneck
            df.loc[idx, "qc_neck_sudden_drop"] = dneck <= -abs(QC_DROP_PER_MIN_THRESHOLD)

        ankle = df.loc[idx, "skin_temp_ankle"]
        if ankle.notna().any():
            dankle = ankle.diff()
            df.loc[idx, "ankle_diff"] = dankle
            df.loc[idx, "qc_ankle_sudden_drop"] = dankle <= -abs(QC_DROP_PER_MIN_THRESHOLD)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and QC skin-temperature files.")
    parser.add_argument("--raw-root", type=Path, default=RAW_SESSION_ROOT, help="Raw session-data root.")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(args.raw_root.rglob("SkinTemp/*_formatted.csv"))
    if not files:
        raise FileNotFoundError(f"No SkinTemp *_formatted.csv files found under {args.raw_root}")

    frames = []
    for f in files:
        try:
            frames.append(standardize_skin_file(f))
        except Exception as e:
            _print(f"[WARN] {f.name}: {e}")

    if not frames:
        raise RuntimeError("No usable SkinTemp files after standardisation.")

    all_rows = pd.concat(frames, ignore_index=True)

    try:
        meta = load_sessions_meta(SESSION_META_CSV)
        clipped = clip_to_session_times(all_rows, meta)
    except Exception as e:
        _print(f"[WARN] Could not apply session clipping: {e}")
        clipped = all_rows

    merged = merge_neck_ankle(clipped)
    out = add_qc_flags(merged)

    out_cols = [
        "part_id", "date", "session_id", "datetime", "time",
        "skin_temp_neck", "skin_temp_ankle",
        "neck_diff", "ankle_diff",
        "qc_neck_sudden_drop", "qc_ankle_sudden_drop",
    ]
    out_cols = [c for c in out_cols if c in out.columns]
    out = out[out_cols].sort_values(["part_id", "date", "session_id", "datetime"])

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Wrote {OUTPUT_CSV} rows={len(out)}")


if __name__ == "__main__":
    main()
