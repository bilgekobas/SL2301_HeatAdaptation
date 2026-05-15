# -*- coding: utf-8 -*-
"""
00_CBT_cleanup_v1_public.py

CBT preprocessing for public-release repository.

Functions:
1. Optional raw CBT clipping/splitting from pill logger exports.
2. Concatenate clipped CBT files.
3. Apply CBT QC flags and write:
   - data/processed/cbt_unflagged.csv
   - data/processed/cbt.csv
   - data/processed/cbt_ok.csv

The public repository does not need to include raw pill logger files. If raw files are
not available, this script will skip raw clipping and can still QC already-clipped files
if data/processed/CBT_CLIPPED exists.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_SESSION_ROOT = DATA_DIR / "raw" / "session_data"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

SESSION_META_CSV = METADATA_DIR / "session_meta.csv"
CBT_CLIPPED_DIR = PROCESSED_DIR / "CBT_CLIPPED"

OUT_ALL = PROCESSED_DIR / "cbt_unflagged.csv"
OUT_FLAGGED = PROCESSED_DIR / "cbt.csv"
OUT_OK = PROCESSED_DIR / "cbt_ok.csv"
PLOTS_DIR = PROCESSED_DIR / "CBT_QC_PLOTS"

LOCAL_TZ = "Europe/Berlin"


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
ALLOWED_DATES = pd.to_datetime([
    "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08", "2023-06-09",
    "2023-06-12", "2023-06-13", "2023-06-14", "2023-06-15", "2023-06-16",
    "2023-06-19", "2023-06-20", "2023-06-21", "2023-06-22", "2023-06-23",
]).normalize()
ALLOWED_SET = set(ALLOWED_DATES)

DEFAULT_SESSION_START = "09:30"
DEFAULT_SESSION_END = "16:30"

ABS_LOW_C = 35.0
WARMUP_HOLD_MIN = 5
DIP_DROP_C = 0.3
DIP_WITHIN_MIN = 9
RECOV_WITHIN_MIN = 15
RECOV_TOL_C = 0.3
SUSTAINED_WIN_MIN = 10
SUSTAINED_DROP_C = 0.3
SMOOTH_WIN_MIN = 3

DROP_COLS = ["SampleNo", "Date", "Time", "CoreTemp", "PillID", "MonitorID", "source_file"]


# ---------------------------------------------------------------------
# Metadata and raw import helpers
# ---------------------------------------------------------------------
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df


def read_session_meta(path: Path = SESSION_META_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Session metadata not found: {path}")

    m = pd.read_csv(path)
    m = normalise_columns(m)

    required_any = ["date", "part_id", "scenario", "condition", "session_id"]
    missing = [c for c in required_any if c not in m.columns]
    if missing:
        raise KeyError(f"session_meta.csv missing required columns: {missing}")

    if "cbt_pill_id" not in m.columns:
        raise KeyError("session_meta.csv must include 'cbt_pill_id' to map CBT pill files.")

    m["cbt_pill_id"] = m["cbt_pill_id"].astype(str).str.strip()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return m


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

    dt = pd.to_datetime(
        df[date_col].astype(str) + " " + df[time_col].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    dt = dt.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")

    out = pd.DataFrame({
        "datetime": dt,
        "pillid": df[pill_col].astype(str).str.strip(),
        "cbt_raw": pd.to_numeric(df[core_col], errors="coerce"),
    })
    return out.dropna(subset=["datetime", "cbt_raw"]).sort_values("datetime").reset_index(drop=True)


def session_clip_mask(ts: pd.Series, session_date: pd.Timestamp,
                      start_hhmm: str = DEFAULT_SESSION_START,
                      end_hhmm: str = DEFAULT_SESSION_END) -> pd.Series:
    start = pd.Timestamp.combine(session_date, pd.Timestamp(start_hhmm).time()).tz_localize(LOCAL_TZ)
    end = pd.Timestamp.combine(session_date, pd.Timestamp(end_hhmm).time()).tz_localize(LOCAL_TZ)
    return (ts >= start) & (ts <= end)


def nice_filename(part_id, session_date: pd.Timestamp, fallback_tag: str = "UNK") -> str:
    date_str = session_date.strftime("%d%m%Y")
    pid = str(part_id) if pd.notna(part_id) else fallback_tag
    return f"{pid}_{date_str}_CBT.csv"


def process_raw_cbt_file(path: Path, meta: pd.DataFrame, out_dir: Path) -> int:
    folder_tag = find_session_tag(path)
    if not folder_tag:
        print(f"[WARN] {path}: cannot find YYMMDD folder; skipping.")
        return 0

    session_day = yymmdd_to_date(folder_tag)
    if session_day not in ALLOWED_SET:
        print(f"[INFO] {path.name}: session {session_day.date()} not in allowed dates; skipping.")
        return 0

    df = read_cbt_semicolon(path)
    df = df.loc[session_clip_mask(df["datetime"], session_day)].copy()
    if df.empty:
        print(f"[INFO] {path.name}: no rows in session window.")
        return 0

    df["date"] = df["datetime"].dt.tz_convert(None).dt.normalize()

    n_written = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for pill_id, chunk in df.groupby("pillid", sort=False):
        day = chunk["date"].iloc[0]
        m = meta[(meta["cbt_pill_id"].astype(str).str.strip() == str(pill_id)) & (meta["date"] == day)]
        if len(m) > 1:
            print(f"[WARN] multiple metadata rows for PillID={pill_id} date={day.date()}, taking first.")

        part_id = m["part_id"].iloc[0] if not m.empty else pd.NA
        scenario = m["scenario"].iloc[0] if not m.empty else pd.NA
        condition = m["condition"].iloc[0] if not m.empty else pd.NA
        session_id = m["session_id"].iloc[0] if not m.empty else pd.NA

        out_df = chunk.copy()
        out_df["part_id"] = part_id
        out_df["scenario"] = scenario
        out_df["condition"] = condition
        out_df["session_id"] = session_id

        out_path = out_dir / nice_filename(part_id, session_day, fallback_tag=f"PILL_{re.sub(r'[^A-Za-z0-9]+', '', str(pill_id))}")
        out_df.to_csv(out_path, index=False)
        print(f"[OK] {path.name} -> {out_path.name} (rows={len(out_df)})")
        n_written += 1

    return n_written


def raw_cleanup(raw_root: Path = RAW_SESSION_ROOT, out_dir: Path = CBT_CLIPPED_DIR) -> None:
    if not raw_root.exists():
        print(f"[SKIP] Raw session root not found: {raw_root}")
        return

    meta = read_session_meta()
    files = sorted(raw_root.rglob("CBT/*_formatted.csv"))
    if not files:
        print(f"[SKIP] no CBT *_formatted.csv files found under {raw_root}")
        return

    total = 0
    for p in files:
        try:
            total += process_raw_cbt_file(p, meta, out_dir)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

    print(f"[DONE] wrote {total} clipped CBT files into {out_dir}")


# ---------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------
def parse_dt(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    return ts.dt.tz_convert(LOCAL_TZ)


def detect_sudden_dips(ts: pd.Series, temp: np.ndarray) -> np.ndarray:
    n = len(temp)
    if n < 3:
        return np.zeros(n, dtype=bool)

    tsec = (ts.view("int64") // 10**9).to_numpy()
    fut_min = np.full(n, np.nan)
    fut_idx = np.full(n, -1, dtype=int)
    j = 0

    for i in range(n):
        while j < n and (tsec[j] - tsec[i]) <= DIP_WITHIN_MIN * 60:
            j += 1
        if j - (i + 1) <= 0:
            continue
        seg = temp[i + 1:j]
        if seg.size == 0:
            continue
        k = i + 1 + seg.argmin()
        fut_min[i] = temp[k]
        fut_idx[i] = k

    drop_mag = temp - fut_min
    is_drop = drop_mag >= DIP_DROP_C

    cold = np.zeros(n, dtype=bool)
    for i, d in enumerate(is_drop):
        if not d:
            continue
        baseline = temp[i]
        k = fut_idx[i]
        if k < 0:
            continue
        end_time = ts.iloc[i] + pd.Timedelta(minutes=RECOV_WITHIN_MIN)
        idxs = np.where((ts >= ts.iloc[k]) & (ts <= end_time))[0]
        recovered = None
        for ridx in idxs:
            if abs(temp[ridx] - baseline) <= RECOV_TOL_C:
                recovered = ridx
                break
        if recovered is not None:
            cold[i:recovered + 1] = True
    return cold


def find_warmup_cutpoint(g: pd.DataFrame) -> int | None:
    s = g.set_index("datetime")["cbt_raw"].copy()
    high = (s >= ABS_LOW_C).astype(int)
    fut_min = high[::-1].rolling(f"{WARMUP_HOLD_MIN}min").min()[::-1]
    ok_points = fut_min[fut_min >= 1.0]
    if ok_points.empty:
        return None
    first_ok_time = ok_points.index.min()
    pos = g.index[g["datetime"] == first_ok_time]
    return int(pos[0]) if len(pos) else None


def interpolate_baseline(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    clean = g["cbt_raw"].copy()
    mask_bad = g["cbt_flag"].ne("ok")
    bad_idx = np.where(mask_bad)[0]

    if bad_idx.size > 0:
        blocks = np.split(bad_idx, np.where(np.diff(bad_idx) > 1)[0] + 1)
        for block in blocks:
            start, end = block[0], block[-1]
            pre_idx = start - 1 if start > 0 else None
            post_idx = end + 1 if end < len(clean) - 1 else None

            if pre_idx is not None and post_idx is not None:
                clean.iloc[start:end + 1] = np.linspace(clean.iloc[pre_idx], clean.iloc[post_idx], end - start + 1)
            elif pre_idx is not None:
                clean.iloc[start:end + 1] = clean.iloc[pre_idx]
            elif post_idx is not None:
                clean.iloc[start:end + 1] = clean.iloc[post_idx]

    g["cbt_clean"] = clean
    return g


def add_flags_one_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("datetime").reset_index(drop=True).copy()
    g["cbt_flag"] = "ok"

    g.loc[g["cbt_raw"] < ABS_LOW_C, "cbt_flag"] = "too_low"
    g.loc[g["cbt_raw"].isna(), "cbt_flag"] = "nan"

    cut_idx = find_warmup_cutpoint(g)
    if cut_idx is not None and cut_idx > 0:
        g.loc[:cut_idx, "cbt_flag"] = "too_low"

    sudden = detect_sudden_dips(g["datetime"], g["cbt_raw"].to_numpy())
    g.loc[sudden, "cbt_flag"] = "cold_water"

    s = g.set_index("datetime")["cbt_raw"]
    baseline = s.rolling(f"{SUSTAINED_WIN_MIN}min", center=True, min_periods=1).median()
    sustained = s < (baseline - SUSTAINED_DROP_C)
    g.loc[sustained.reindex(g["datetime"]).fillna(False).values, "cbt_flag"] = "cold_water"

    g = interpolate_baseline(g)

    if SMOOTH_WIN_MIN and SMOOTH_WIN_MIN > 0:
        s2 = g.set_index("datetime")["cbt_clean"].rolling(f"{SMOOTH_WIN_MIN}min", center=True, min_periods=1).median()
        g["cbt_clean_smooth"] = s2.reindex(g["datetime"]).values
    else:
        g["cbt_clean_smooth"] = g["cbt_clean"]

    return g


def plot_group(g: pd.DataFrame, out_path: Path, title: str) -> None:
    g = g.sort_values("datetime")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(g["datetime"], g["cbt_raw"], linewidth=1.5, label="CBT raw")
    ax.plot(g["datetime"], g["cbt_clean_smooth"], linewidth=1.2, linestyle="--", label="CBT clean")

    bad = g["cbt_flag"].ne("ok").to_numpy()
    dt = g["datetime"].to_numpy()
    if bad.any():
        start = None
        for i, is_bad in enumerate(bad):
            if is_bad and start is None:
                start = i
            if (not is_bad and start is not None) or (is_bad and i == len(bad) - 1):
                end = i if not is_bad else i
                ax.axvspan(dt[start], dt[end], alpha=0.15, ec=None, fc="0.5")
                start = None

    ax.set_ylabel("CBT (°C)")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def concatenate_clipped(cbt_clipped_dir: Path = CBT_CLIPPED_DIR, out_path: Path = OUT_ALL) -> pd.DataFrame:
    files = sorted([f for f in cbt_clipped_dir.glob("P*_CBT.csv") if not f.name.endswith("_events.csv")])
    if not files:
        raise FileNotFoundError(f"No CBT files found under {cbt_clipped_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        drop_now = [c for c in DROP_COLS if c in df.columns]
        if drop_now:
            df = df.drop(columns=drop_now)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    sort_cols = [c for c in ["part_id", "session_id", "datetime"] if c in all_df.columns]
    if sort_cols:
        all_df = all_df.sort_values(sort_cols).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} with {len(all_df)} rows from {len(files)} files.")
    return all_df


def apply_cbt_qc(df: pd.DataFrame, out_flagged: Path = OUT_FLAGGED, out_ok: Path = OUT_OK) -> pd.DataFrame:
    for c in ["datetime", "cbt_raw"]:
        if c not in df.columns:
            raise KeyError(f"Input missing '{c}' column.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["datetime"] = parse_dt(df["datetime"])

    if "part_id" not in df.columns:
        df["part_id"] = "UNK"
    if "session_id" not in df.columns:
        if "date" in df.columns:
            df["session_id"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("D%Y%m%d")
        else:
            df["session_id"] = "UNK"

    chunks = []
    for (pid, sid), g in df.groupby(["part_id", "session_id"], sort=True):
        g2 = add_flags_one_group(g)
        chunks.append(g2)
        plot_name = f"{str(pid).replace('/', '-')}_{str(sid).replace('/', '-')}_CBT_QC.png"
        plot_group(g2, PLOTS_DIR / plot_name, title=f"{pid} – {sid}")

    flagged = pd.concat(chunks, ignore_index=True)
    flagged.to_csv(out_flagged, index=False)
    print(f"[OK] Wrote {out_flagged} with {len(flagged)} rows.")

    ok_only = flagged[flagged["cbt_flag"] == "ok"].copy()
    ok_only.to_csv(out_ok, index=False)
    print(f"[OK] Wrote {out_ok} with {len(ok_only)} usable rows.")

    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description="CBT preprocessing and QC.")
    parser.add_argument("--skip-raw", action="store_true", help="Skip raw CBT clipping/splitting.")
    parser.add_argument("--raw-root", type=Path, default=RAW_SESSION_ROOT, help="Raw session-data root.")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_raw:
        raw_cleanup(args.raw_root, CBT_CLIPPED_DIR)

    if CBT_CLIPPED_DIR.exists() and any(CBT_CLIPPED_DIR.glob("P*_CBT.csv")):
        df = concatenate_clipped(CBT_CLIPPED_DIR, OUT_ALL)
    elif OUT_ALL.exists():
        print(f"[INFO] Using existing concatenated CBT file: {OUT_ALL}")
        df = pd.read_csv(OUT_ALL)
    else:
        raise FileNotFoundError("No clipped CBT files and no existing cbt_unflagged.csv found.")

    apply_cbt_qc(df, OUT_FLAGGED, OUT_OK)


if __name__ == "__main__":
    main()
