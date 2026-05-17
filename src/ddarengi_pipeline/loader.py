import glob
import os
from typing import List

import pandas as pd


def _try_read_csv(path: str, nrows=None):
    encodings = ["cp949", "utf-8", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except Exception:
            continue
    # last resort
    return pd.read_csv(path, engine="python", error_bad_lines=False)


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # clean up column names: strip whitespace and newlines
    df.columns = df.columns.str.strip().str.replace('\n', ' ')
    
    cols = list(df.columns)
    col_map = {}
    lower_cols = [c.lower() for c in cols]

    def find_containing(subs: List[str]):
        for i, c in enumerate(lower_cols):
            for s in subs:
                if s in c:
                    return cols[i]
        return None

    # heuristics for common Korean column names
    start_time_col = find_containing(["대여일시", "대여일", "대여시간", "rent", "start", "rental"])
    end_time_col = find_containing(["반납일시", "반납일", "반납시간", "return", "end"])
    # priority: look for ID columns first, then fallback to 번호 columns
    start_station_col = find_containing(["대여대여소id", "대여소id", "대여대여소 id", "대여대여소번호", "대여소번호", "start", "origin"])
    end_station_col = find_containing(["반납대여소id", "반납대여소 id", "반납대여소번호", "반납대여소명", "return", "destination"])

    # fallback: look for any column containing '대여소' and use ID variants
    if start_station_col is None or end_station_col is None:
        id_cols = [c for c in cols if "id" in c.lower()]
        station_related = [c for c in cols if "대여소" in c]
        
        if start_station_col is None and id_cols:
            # prefer columns with "대여" but not "반납"
            for c in id_cols:
                if "대여" in c and "반납" not in c:
                    start_station_col = c
                    break
            if start_station_col is None:
                start_station_col = id_cols[0]
        
        if end_station_col is None and len(id_cols) > 1:
            for c in id_cols:
                if "반납" in c:
                    end_station_col = c
                    break
            if end_station_col is None:
                end_station_col = id_cols[1] if len(id_cols) > 1 else id_cols[0]

    # map to canonical names
    if start_time_col:
        col_map[start_time_col] = "start_time"
    if end_time_col:
        col_map[end_time_col] = "end_time"
    if start_station_col:
        col_map[start_station_col] = "start_station_id"
    if end_station_col:
        col_map[end_station_col] = "end_station_id"

    df = df.rename(columns=col_map)

    # parse datetime columns
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    return df


def load_rental_history_from_dir(ddarengi_dir: str, nrows=None) -> pd.DataFrame:
    """Load and concatenate all rental history CSVs from a directory.

    Returns a DataFrame with canonical columns: `start_time`, `end_time`,
    `start_station_id`, `end_station_id` when possible.
    """
    pattern = os.path.join(ddarengi_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {ddarengi_dir}")

    dfs = []
    for f in files:
        try:
            df = _try_read_csv(f, nrows=nrows)
        except Exception as e:
            continue
        df = _canonicalize_columns(df)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # ensure we have at least a start_time column
    if "start_time" not in combined.columns:
        raise ValueError("Cannot detect start_time column in the provided CSVs. Please inspect files or provide column mapping.")

    combined = combined.sort_values("start_time").reset_index(drop=True)
    return combined
