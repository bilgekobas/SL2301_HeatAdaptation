# -*- coding: utf-8 -*-
"""
00_HRV_v2_public.py

ECG -> RR/HRV preprocessing using repository-relative paths.

Primary outputs:
    data/processed/HRV_OUT/*_RR.csv
    data/processed/HRV_OUT/*_HRV_5min.csv
    data/processed/hrv_no_sedentary.csv
    data/processed/hrv.csv

Raw ECG files are expected under:
    data/raw/session_data/**/ECG/**/*_Calibrated_SD.csv

Session metadata:
    data/metadata/session_meta.csv

Notes:
- Raw ECG is not expected to be stored in GitHub.
- The large rescue/debug blocks from the working script were removed for public clarity.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import re
import csv
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal

try:
    import neurokit2 as nk
except Exception:
    nk = None

try:
    from numpy import trapezoid as _trapint
except Exception:
    _trapint = np.trapz


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_SESSION_ROOT = DATA_DIR / "raw" / "session_data"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

SESSION_META_CSV = METADATA_DIR / "session_meta.csv"
HRV_OUT = PROCESSED_DIR / "HRV_OUT"
OUT_ALL_HRV = PROCESSED_DIR / "hrv_no_sedentary.csv"
ACT_5MIN = PROCESSED_DIR / "activity_5min.csv"
OUT_WITH_SED = PROCESSED_DIR / "hrv.csv"

LOCAL_TZ = "Europe/Berlin"
VALID_EXT = (".csv",)
TIMESTAMP_SUFFIX = "_Timestamp_Unix_CAL"
CTX_SUFFIXES = {"temp_bmp280": "_Temperature_BMP280_CAL"}

LEAD_ORDER_DEFAULT = ["LL-RA", "LA-RA", "LL-LA", "Vx-RL"]
LEAD_ORDER_OVERRIDE = {
    "SL092": ["LA-RA", "LL-RA", "LL-LA", "Vx-RL"],
    "SL097": ["LA-RA", "LL-RA", "LL-LA", "Vx-RL"],
}

warnings.filterwarnings("ignore", message="DFA_alpha2 related indices will not be calculated.*")


def _device_suffix_from_ecg_id(ecg_id: str) -> str:
    m = re.search(r"_([A-Z0-9]{4,})$", str(ecg_id).strip().upper())
    return m.group(1) if m else ""


def _linear_detrend(y: np.ndarray) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    m, b = np.polyfit(x, y, 1)
    return y - (m * x + b)


def _bandpower(f, Pxx, fmin, fmax):
    sel = (f >= fmin) & (f < fmax)
    if not np.any(sel):
        return 0.0
    return float(_trapint(Pxx[sel], f[sel]))


def _hrv_freq_manual_welch(peaks_idx, timestamps_ms, fs=4.0):
    try:
        from scipy.signal import welch
    except Exception:
        return None

    if len(peaks_idx) < 8:
        return None

    t = np.asarray(timestamps_ms.iloc[peaks_idx], dtype=float) / 1000.0
    rr = np.diff(t)
    if np.any(~np.isfinite(rr)) or rr.size < 64:
        return None

    tm = 0.5 * (t[1:] + t[:-1])
    t_even = np.arange(tm[0], tm[-1], 1.0 / fs)
    if t_even.size < 128:
        return None

    rr_even = np.interp(t_even, tm, rr)
    rr_even = _linear_detrend(rr_even)

    nperseg = min(256, int(2 ** np.floor(np.log2(rr_even.size))))
    if nperseg < 64:
        return None

    f, pxx = welch(rr_even, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend=False)
    vlf = _bandpower(f, pxx, 0.0033, 0.04)
    lf = _bandpower(f, pxx, 0.04, 0.15)
    hf = _bandpower(f, pxx, 0.15, 0.40)
    tp = _bandpower(f, pxx, 0.0033, 0.40)

    lfn = 100.0 * lf / (lf + hf) if (lf + hf) > 0 else np.nan
    hfn = 100.0 * hf / (lf + hf) if (lf + hf) > 0 else np.nan
    lf_hf = lf / hf if hf > 0 else np.nan

    hf_band = (f >= 0.15) & (f < 0.40)
    hf_idx = np.where(hf_band & np.isfinite(pxx))[0]
    hf_peak_hz = float(f[hf_idx[np.argmax(pxx[hf_idx])]]) if hf_idx.size else np.nan

    return pd.DataFrame({
        "HRV_VLF": [vlf * 1e6], "HRV_LF": [lf * 1e6], "HRV_HF": [hf * 1e6], "HRV_TP": [tp * 1e6],
        "HRV_LFn": [lfn], "HRV_HFn": [hfn], "HRV_LFHF": [lf_hf],
        "HF_peak_Hz": [hf_peak_hz], "Resp_rate_bpm": [hf_peak_hz * 60.0 if np.isfinite(hf_peak_hz) else np.nan],
        "freq_method": ["manual_welch_4Hz"],
    })


def detect_sep_and_skiprows(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().lstrip("\ufeff").rstrip("\n\r")

    if first.lower().startswith("sep="):
        raw = first[4:].strip()
        sep = "\t" if raw in (r"\t", "\\t") else (raw if raw else ",")
        return sep, 1, 1, True

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            probe = ""
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    probe = line
                    break
        sep = csv.Sniffer().sniff(probe, delimiters=",;\t|").delimiter if probe else ","
        return sep, 0, 0, False
    except Exception:
        return ",", 0, 0, False


def read_header(path, sep, skip_h):
    return [c.strip() for c in pd.read_csv(path, sep=sep, nrows=0, header=0, skiprows=skip_h).columns.tolist()]


BAD_TOKENS = ("ECGtoHR", "BPM", "HeartRate", "HRV", "RR", "IBI", "Quality", "Confidence")


def find_cols(csv_path: str, sep: str, skiprows_header, skiprows_data, sensor_id: str = ""):
    cols = read_header(csv_path, sep, skiprows_header)

    ecg_all = [c for c in cols if re.search(r"ECG", c, flags=re.I)]
    ecg_all = [c for c in ecg_all if not any(bt in c for bt in BAD_TOKENS)]
    if not ecg_all:
        raise ValueError("No ECG-like columns found.")

    dev = _device_suffix_from_ecg_id(sensor_id)
    order = LEAD_ORDER_OVERRIDE.get(dev, LEAD_ORDER_DEFAULT)
    pref_suffixes = [f"_ECG_{tag}_24BIT_CAL" for tag in order]

    def score_ecg(name: str):
        try:
            pref_rank = next(i for i, suf in enumerate(pref_suffixes) if name.endswith(suf))
        except StopIteration:
            pref_rank = len(pref_suffixes) + 1
        cal_rank = 0 if name.endswith("_24BIT_CAL") else (0 if name.endswith("_CAL") else 1)
        lead_like = 0 if re.search(r"ECG_[A-Z]+-[A-Z]+_", name) else 1
        return pref_rank, lead_like, cal_rank, len(name)

    ecg_sorted = sorted(ecg_all, key=score_ecg)
    ecg_col = ecg_sorted[0]

    ts_candidates = [c for c in cols if c.endswith(TIMESTAMP_SUFFIX)]
    if not ts_candidates:
        ts_candidates = [c for c in cols if re.search(r"Timestamp.*Unix.*_CAL$", c)]
    if not ts_candidates:
        ts_candidates = [c for c in cols if re.search(r"Timestamp", c, flags=re.I)]
    if not ts_candidates:
        ts_candidates = [c for c in cols if re.search(r"(Time|Datetime|DateTime)", c, flags=re.I)]
    if not ts_candidates:
        raise ValueError("No timestamp-like column found.")

    return ts_candidates[0], ecg_col, cols


def find_optional_ctx_cols(all_cols):
    mapping = {}
    for key, suffix in CTX_SUFFIXES.items():
        cand = [c for c in all_cols if c.endswith(suffix)]
        mapping[key] = cand[0] if cand else None
    return mapping


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


def load_signal(csv_path, sep, skiprows_data, timestamp_col, ecg_col, ctx_map):
    usecols = [timestamp_col, ecg_col] + [c for c in ctx_map.values() if c]
    df = pd.read_csv(csv_path, sep=sep, header=0, skiprows=skiprows_data, usecols=usecols)
    ren = {timestamp_col: "timestamp_raw", ecg_col: "ecg_mv"}
    for key, col in ctx_map.items():
        if col:
            ren[col] = key
    df = df.rename(columns=ren)

    df["timestamp_unix_ms"] = _parse_timestamp_to_ms(df["timestamp_raw"])
    keep = ["timestamp_unix_ms", "ecg_mv"] + [k for k in ctx_map.keys() if k in df.columns]
    return df[keep].dropna(subset=["timestamp_unix_ms", "ecg_mv"]).reset_index(drop=True)


def estimate_sr_from_timestamps(ts_ms_series: pd.Series) -> int:
    ts = pd.to_numeric(ts_ms_series, errors="coerce").dropna().astype(np.float64)
    if len(ts) < 10:
        return 128
    diffs = np.diff(ts)
    diffs = diffs[(diffs > 0) & (diffs < np.nanpercentile(diffs, 99.9))]
    if len(diffs) == 0:
        return 128
    median_ms = float(np.nanmedian(diffs))
    sr_est = int(round(1000.0 / median_ms)) if median_ms > 0 else 128
    return max(1, sr_est)


def compute_rr_and_peaks(ecg_mv, sr):
    if nk is None:
        raise ImportError("neurokit2 is required.")

    ecg_cleaned = nk.ecg_clean(ecg_mv, sampling_rate=sr, method="neurokit")
    for method in ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010"]:
        try:
            _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sr, method=method)
            peaks_idx = rpeaks.get("ECG_R_Peaks", None)
            if peaks_idx is not None and len(peaks_idx) >= 2:
                rr_ms = np.diff(peaks_idx) * (1000.0 / sr)
                return peaks_idx, rr_ms
        except Exception:
            continue

    raise ValueError("No R-peaks detected.")


def _hrv_freq_safe(peaks_idx, timestamps_ms, sr, mode="auto"):
    if mode == "none":
        return None
    if nk is None:
        return _hrv_freq_manual_welch(peaks_idx, timestamps_ms)

    order = ["lomb", "fft", "welch", "ar"] if mode == "auto" else [mode]
    for m in order:
        try:
            df = nk.hrv_frequency({"ECG_R_Peaks": np.asarray(peaks_idx, dtype=int)}, sampling_rate=sr, method=m, show=False)
            df = df.copy()
            df["freq_method"] = m
            return df
        except Exception:
            continue

    return _hrv_freq_manual_welch(peaks_idx, timestamps_ms)


def hrv_rolling_windows(peaks_idx, timestamps_ms, window_sec=300, step_sec=60, sr=128, ctx_df=None, freq_mode="auto", pbar=None):
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    peaks_times_ms = timestamps_ms.iloc[peaks_idx].to_numpy()

    start_time = float(peaks_times_ms[0])
    end_time = float(peaks_times_ms[-1])
    window_ms = window_sec * 1000.0
    step_ms = step_sec * 1000.0

    if end_time - start_time < window_ms:
        return pd.DataFrame()

    n_windows = int((end_time - start_time - window_ms) // step_ms) + 1
    rows = []
    t0 = start_time

    for _ in range(n_windows):
        t1 = t0 + window_ms
        sel = (peaks_times_ms >= t0) & (peaks_times_ms < t1)
        p_win = peaks_idx[sel]

        if len(p_win) >= 2:
            try:
                parts = [nk.hrv_time({"ECG_R_Peaks": p_win}, sampling_rate=sr, show=False)]
                freq_df = _hrv_freq_safe(p_win, timestamps_ms, sr, mode=freq_mode)
                if freq_df is not None:
                    parts.append(freq_df)
                try:
                    parts.append(nk.hrv_nonlinear({"ECG_R_Peaks": p_win}, sampling_rate=sr, show=False))
                except Exception:
                    pass

                row = pd.concat(parts, axis=1)
                rr_win_ms = np.diff(p_win) * (1000.0 / sr)
                row["bpm_mean"] = 60000.0 / float(np.nanmean(rr_win_ms)) if rr_win_ms.size else np.nan
                row.insert(0, "window_start_ms", t0)
                row.insert(1, "window_end_ms", t1)

                if ctx_df is not None and not ctx_df.empty:
                    m = (ctx_df["timestamp_unix_ms"] >= t0) & (ctx_df["timestamp_unix_ms"] < t1)
                    for col in [c for c in ctx_df.columns if c != "timestamp_unix_ms"]:
                        row[col + "_mean"] = ctx_df.loc[m, col].mean()

                rows.append(row)
            except Exception:
                pass

        if pbar is not None:
            pbar.update(1)
        t0 += step_ms

    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df


def _parse_date_any(x):
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        s = str(x)
        if len(s) == 6 and s.isdigit():
            dt = pd.to_datetime("20" + s, format="%Y%m%d", errors="coerce")
    return None if pd.isna(dt) else dt.normalize()


def _combine_date_time_to_ms(date_ts: pd.Timestamp, time_val):
    if pd.isna(date_ts):
        return None
    t_only = pd.to_datetime(str(time_val), errors="coerce")
    if pd.isna(t_only):
        return None
    dt = pd.Timestamp.combine(date_ts.date(), t_only.to_pydatetime().time())
    dt = dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    return float(dt.tz_convert("UTC").value // 10**6)


def _clean_session_id(x) -> str:
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def load_metadata(meta_path: Path = SESSION_META_CSV) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    df = _norm_cols(df)

    for col in ["date", "ecg_id", "session_start", "session_end"]:
        if col not in df.columns:
            raise ValueError(f"Metadata missing required column: {col}")

    df["date_norm"] = df["date"].apply(_parse_date_any)
    df["ecg_id_norm"] = df["ecg_id"].astype(str).str.strip().str.upper()
    df["ignore"] = pd.to_numeric(df.get("ignore", 0), errors="coerce").fillna(0).astype(int)
    df["start_ms"] = df.apply(lambda r: _combine_date_time_to_ms(r["date_norm"], r["session_start"]), axis=1)
    df["end_ms"] = df.apply(lambda r: _combine_date_time_to_ms(r["date_norm"], r["session_end"]), axis=1)
    df["session_id_norm"] = df.get("session_id", pd.Series(pd.NA, index=df.index)).apply(_clean_session_id)
    df["part_id_norm"] = df.get("part_id", pd.Series(pd.NA, index=df.index)).astype(str).str.strip()

    return df


def extract_sensor_id_from_filename(path_str: str) -> str:
    base = os.path.basename(path_str)
    m = re.search(r"_(S[0-9A-Z]+_[0-9A-Z]+)_", base)
    if m:
        return m.group(1).upper()
    m = re.search(r"_(S[0-9A-Z]+_[0-9A-Z]+)(?:_|\.|$)", base)
    return m.group(1).upper() if m else ""


def extract_session_id_from_filename(path_str: str) -> str:
    base = os.path.basename(path_str)
    m = re.search(r"Session(\d+)", base, flags=re.I)
    return m.group(1) if m else ""


def parse_date_from_path(path_str: str):
    for p in Path(path_str).parts:
        if re.fullmatch(r"\d{6}", p or ""):
            return pd.to_datetime("20" + p, format="%Y%m%d", errors="coerce")
    return None


def match_metadata_row(meta_df: pd.DataFrame, csv_path: str, sensor_id: str):
    date_guess = parse_date_from_path(csv_path)
    date_norm = date_guess.normalize() if date_guess is not None else None
    session_id_guess = extract_session_id_from_filename(csv_path)
    ecg_norm = sensor_id.upper()

    m = meta_df[(meta_df["ecg_id_norm"] == ecg_norm) & (meta_df["date_norm"] == date_norm)].copy()
    if session_id_guess:
        m2 = m[m["session_id_norm"] == session_id_guess]
        if len(m2) == 1:
            m = m2

    if m.empty:
        return None

    row = m.iloc[0]
    return {
        "part_id": row.get("part_id_norm", pd.NA),
        "date_str": None if pd.isna(row["date_norm"]) else row["date_norm"].strftime("%Y-%m-%d"),
        "session_id": row.get("session_id_norm", ""),
        "start_ms": row.get("start_ms", None),
        "end_ms": row.get("end_ms", None),
        "ignore": int(row.get("ignore", 0)),
    }


def is_under_ecg_folder(dirpath: str) -> bool:
    return "ecg" in [p.lower() for p in Path(dirpath).parts]


def should_process_filename(filename: str) -> bool:
    return filename.lower().endswith("_calibrated_sd.csv")


def process_file(csv_path, out_dir, sampling_rate=None, do_5min=True, window_sec=300, step_sec=60,
                 freq_mode="auto", meta_df: pd.DataFrame | None = None):
    os.makedirs(out_dir, exist_ok=True)

    sep, skiprows_header, skiprows_data, _ = detect_sep_and_skiprows(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    rr_out = os.path.join(out_dir, f"{base}_RR.csv")
    hrv_5min_out = os.path.join(out_dir, f"{base}_HRV_5min.csv")

    sensor_id = extract_sensor_id_from_filename(csv_path)
    timestamp_col, ecg_col, all_cols = find_cols(csv_path, sep, skiprows_header, skiprows_data, sensor_id=sensor_id)
    ctx_map = find_optional_ctx_cols(all_cols)
    meta_info = match_metadata_row(meta_df, csv_path, sensor_id) if meta_df is not None else None

    part_id = None
    date_str = None
    session_id = ""

    df = load_signal(csv_path, sep, skiprows_data, timestamp_col, ecg_col, ctx_map)

    if meta_info is not None:
        if int(meta_info.get("ignore", 0)) == 1:
            raise RuntimeError("Metadata flag ignore==1; skipping file.")

        part_id = meta_info.get("part_id")
        date_str = meta_info.get("date_str")
        session_id = meta_info.get("session_id", "")

        s_ms = meta_info.get("start_ms")
        e_ms = meta_info.get("end_ms")
        if isinstance(s_ms, (int, float)) and isinstance(e_ms, (int, float)):
            df = df.loc[(df["timestamp_unix_ms"] >= s_ms) & (df["timestamp_unix_ms"] <= e_ms)].reset_index(drop=True)
            if len(df) < 1000:
                raise RuntimeError("After clipping to session window, not enough data points.")
    else:
        tqdm.write(f"[WARN] No metadata match for {os.path.basename(csv_path)}; processing full file.")

    sr = int(sampling_rate) if sampling_rate is not None else estimate_sr_from_timestamps(df["timestamp_unix_ms"])
    if sr <= 1:
        sr = 128

    peaks_idx, rr_ms = compute_rr_and_peaks(df["ecg_mv"].to_numpy(), sr=sr)
    rr = np.diff(peaks_idx) * (1000.0 / sr)
    good = (rr >= 300.0) & (rr <= 2000.0)
    if good.sum() >= 2:
        peaks_idx = np.concatenate([peaks_idx[:1], peaks_idx[1:][good]])
        rr_ms = np.diff(peaks_idx) * (1000.0 / sr)

    beat_times_ms = df["timestamp_unix_ms"].iloc[peaks_idx].to_numpy()

    rr_df = pd.DataFrame({"beat_time_ms": beat_times_ms[1:].astype(np.float64)})
    ctx_cols = [c for c in ["temp_bmp280"] if c in df.columns]
    if ctx_cols:
        ctx_ts = df[["timestamp_unix_ms"] + ctx_cols].dropna(subset=["timestamp_unix_ms"]).sort_values("timestamp_unix_ms")
        rr_df = pd.merge_asof(rr_df.sort_values("beat_time_ms"), ctx_ts,
                              left_on="beat_time_ms", right_on="timestamp_unix_ms", direction="nearest")
        rr_df = rr_df.drop(columns=["timestamp_unix_ms"], errors="ignore")

    rr_df.insert(0, "ecg_id", sensor_id)
    if part_id is not None:
        rr_df.insert(1, "part_id", part_id)
    if date_str is not None:
        rr_df.insert(2, "date", date_str)
    if session_id:
        rr_df.insert(3, "session_id", session_id)
    rr_df["rr_ms"] = rr_ms.astype(np.float32)
    rr_df.to_csv(rr_out, index=False)

    if do_5min:
        ctx_df = df[["timestamp_unix_ms"] + ctx_cols].copy() if ctx_cols else None
        start_ms = float(df["timestamp_unix_ms"].iloc[0])
        end_ms = float(df["timestamp_unix_ms"].iloc[-1])
        n_windows = int((end_ms - start_ms - window_sec * 1000) // (step_sec * 1000)) + 1 if end_ms - start_ms >= window_sec * 1000 else 0

        pbar = tqdm(total=max(0, n_windows), desc=f"5-min HRV: {os.path.basename(csv_path)}", unit="win", leave=False)
        rolling_df = hrv_rolling_windows(peaks_idx, df["timestamp_unix_ms"], window_sec, step_sec, sr, ctx_df, freq_mode, pbar)
        pbar.close()

        if not rolling_df.empty:
            rolling_df.insert(0, "file", os.path.basename(csv_path))
            rolling_df.insert(1, "ecg_id", sensor_id)
            if part_id is not None:
                rolling_df.insert(2, "part_id", part_id)
            if date_str is not None:
                rolling_df.insert(3, "date", date_str)
            if session_id:
                rolling_df.insert(4, "session_id", session_id)
            rolling_df.to_csv(hrv_5min_out, index=False)

    return {"file": csv_path, "sampling_rate": sr, "rr_csv": rr_out, "rolling_csv": hrv_5min_out if do_5min else None}


def scan_and_process(root_dir, out_dir, sampling_rate=None, do_5min=True, window_sec=300, step_sec=60,
                     freq_mode="auto", meta_path: Path = SESSION_META_CSV):
    meta_df = load_metadata(meta_path) if meta_path and Path(meta_path).exists() else None

    targets = []
    for dirpath, _, filenames in os.walk(root_dir):
        if not is_under_ecg_folder(dirpath):
            continue
        for fn in filenames:
            if fn.lower().endswith(VALID_EXT) and should_process_filename(fn):
                targets.append(os.path.join(dirpath, fn))

    results = []
    for fpath in tqdm(targets, desc="Files", unit="file"):
        try:
            tqdm.write(f"[START] {fpath}")
            res = process_file(fpath, out_dir, sampling_rate, do_5min, window_sec, step_sec, freq_mode, meta_df=meta_df)
            results.append(res)
            tqdm.write(f"[OK] {fpath} -> SR={res['sampling_rate']} Hz")
        except Exception as e:
            tqdm.write(f"[ERROR] {fpath}: {e}")

    if results:
        pd.DataFrame(results).to_csv(Path(out_dir) / "manifest.csv", index=False)
    return results


def concat_hrv_outputs(hrv_out: Path = HRV_OUT, out_path: Path = OUT_ALL_HRV) -> pd.DataFrame:
    files = sorted(hrv_out.glob("*_HRV_5min.csv"))
    if not files:
        raise FileNotFoundError(f"No *_HRV_5min.csv files found under {hrv_out}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} rows={len(all_df)}")
    return all_df


def prep_activity_5min(path: Path = ACT_5MIN) -> pd.DataFrame:
    a = pd.read_csv(path)
    a.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in a.columns]

    req = ["ecg_id", "minute_start", "valid_for_hrv"]
    missing = [c for c in req if c not in a.columns]
    if missing:
        raise KeyError(f"Activity file missing columns: {missing}")

    dt = pd.to_datetime(a["minute_start"], utc=False, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    a["datetime_min"] = dt.dt.round("min")

    a["ecg_id"] = a["ecg_id"].astype(str).str.upper().str.strip()
    a["date"] = a["datetime_min"].dt.tz_convert(None).dt.normalize()
    a = a.rename(columns={"valid_for_hrv": "sedentary", "activity_step_count_cal": "step_count"})

    keep = ["ecg_id", "date", "datetime_min", "sedentary"]
    if "step_count" in a.columns:
        keep.append("step_count")

    return a[keep].drop_duplicates(subset=["ecg_id", "date", "datetime_min"])


def add_sedentary_to_hrv(hrv_path: Path = OUT_ALL_HRV, activity_path: Path = ACT_5MIN, out_path: Path = OUT_WITH_SED) -> pd.DataFrame:
    hrv = pd.read_csv(hrv_path)
    hrv.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in hrv.columns]

    # Build datetime_start if only ms window starts are present
    if "datetime_start" not in hrv.columns and "window_start_ms" in hrv.columns:
        hrv["datetime_start"] = pd.to_datetime(hrv["window_start_ms"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ)

    needed = ["ecg_id", "datetime_start", "date"]
    missing = [c for c in needed if c not in hrv.columns]
    if missing:
        raise KeyError(f"HRV file missing columns: {missing}")

    hrv["datetime_start"] = pd.to_datetime(hrv["datetime_start"], errors="coerce")
    if getattr(hrv["datetime_start"].dt, "tz", None) is None:
        hrv["datetime_start"] = hrv["datetime_start"].dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT")
    hrv["datetime_min"] = hrv["datetime_start"].dt.round("min")
    hrv["ecg_id"] = hrv["ecg_id"].astype(str).str.upper().str.strip()
    hrv["date"] = pd.to_datetime(hrv["date"], errors="coerce").dt.tz_localize(None).dt.normalize()

    act5 = prep_activity_5min(activity_path)
    merged = hrv.merge(
        act5.rename(columns={"datetime_min": "dt_merge"}),
        how="left",
        left_on=["ecg_id", "date", "datetime_min"],
        right_on=["ecg_id", "date", "dt_merge"],
    ).drop(columns=["dt_merge"])

    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}. Sedentary coverage: {merged['sedentary'].notna().mean():.1%}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG to HRV preprocessing.")
    parser.add_argument("--raw-root", type=Path, default=RAW_SESSION_ROOT)
    parser.add_argument("--out", type=Path, default=HRV_OUT)
    parser.add_argument("--sr", type=int, default=128)
    parser.add_argument("--skip-ecg-processing", action="store_true")
    parser.add_argument("--skip-sedentary-merge", action="store_true")
    args = parser.parse_args()

    if nk is None and not args.skip_ecg_processing:
        raise ImportError("neurokit2 is required. Install with: pip install neurokit2")

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_ecg_processing:
        scan_and_process(args.raw_root, args.out, args.sr, True, 300, 60, "auto", SESSION_META_CSV)

    if args.out.exists() and any(args.out.glob("*_HRV_5min.csv")):
        concat_hrv_outputs(args.out, OUT_ALL_HRV)

    if not args.skip_sedentary_merge and OUT_ALL_HRV.exists() and ACT_5MIN.exists():
        add_sedentary_to_hrv(OUT_ALL_HRV, ACT_5MIN, OUT_WITH_SED)


if __name__ == "__main__":
    main()
