import glob
import os
from typing import List

import pandas as pd


def _try_read_csv(path: str, nrows=None, usecols=None):
    encodings = ["cp949", "utf-8", "euc-kr"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, usecols=usecols)
        except Exception:
            continue
    # last resort
    return pd.read_csv(path, engine="python", error_bad_lines=False, nrows=nrows)


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


def load_rental_history_from_dir(ddarengi_dir: str, nrows=None, max_rows=None, sample_frac=None) -> pd.DataFrame:
    """Load and concatenate rental history CSVs from a directory.

    Args:
        ddarengi_dir: directory containing CSV rental history files.
        nrows: read only the first nrows of each file.
        max_rows: stop loading once this many rows are collected across all files.
        sample_frac: if set, randomly sample this fraction from each file.

    Returns a DataFrame with canonical columns: `start_time`, `end_time`,
    `start_station_id`, `end_station_id` when possible.
    """
    pattern = os.path.join(ddarengi_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {ddarengi_dir}")

    print(f"Found {len(files)} CSV files in {ddarengi_dir}")
    
    dfs = []
    total_rows = 0
    for i, f in enumerate(files):
        if max_rows is not None and total_rows >= max_rows:
            break

        try:
            print(f"  Loading file {i+1}/{len(files)}: {os.path.basename(f)}")
            file_nrows = nrows
            if file_nrows is None and max_rows is not None:
                file_nrows = max_rows - total_rows
            df = _try_read_csv(f, nrows=file_nrows)
        except Exception as e:
            print(f"    WARNING: Failed to load {f}: {e}")
            continue

        df = _canonicalize_columns(df)
        if sample_frac is not None and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=42)

        if max_rows is not None and len(df) > max_rows - total_rows:
            df = df.iloc[: max_rows - total_rows]

        dfs.append(df)
        total_rows += len(df)
        print(f"    ✓ Loaded {len(df)} records (total {total_rows})")

    if not dfs:
        raise ValueError("No rental data could be loaded from the provided CSV files.")

    print("Concatenating dataframes...")
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # ensure we have at least a start_time column
    if "start_time" not in combined.columns:
        raise ValueError("Cannot detect start_time column in the provided CSVs. Please inspect files or provide column mapping.")

    print("Sorting by start_time...")
    combined = combined.sort_values("start_time").reset_index(drop=True)
    print(f"Total records loaded: {len(combined)}")
    return combined
