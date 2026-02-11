# ML/dataio.py
# Purpose: load ATP match CSVs by year, standardize, parse dates, filter obvious junk,
# and (optionally) drop post-match leakage columns (w_*, l_*, score, minutes, etc.)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class DataIOConfig:
    data_dir: Path
    filename_template: str = "atp_matches_{year}.csv"
    drop_leakage: bool = True
    remove_walkovers: bool = True
    keep_only_singles: bool = False  # if your dataset supports it (often it doesn't)


# Minimal columns we need to proceed
REQUIRED_COLS = ("tourney_date", "winner_id", "loser_id")

# Obvious post-match columns
LEAKAGE_EXACT = {"score", "minutes"}
LEAKAGE_PREFIXES = ("w_", "l_")  # post-match stats prefixes in many ATP datasets

# Common optional columns (used if present)
OPTIONAL_ORDER_COLS = ("match_num", "tourney_id")


# ----------------------------
# Public API
# ----------------------------

def path_for_year(cfg: DataIOConfig, year: int) -> Path:
    """Return the expected file path for a given year."""
    return cfg.data_dir / cfg.filename_template.format(year=year)


def read_year(cfg: DataIOConfig, year: int) -> pd.DataFrame:
    """
    Read a single year's CSV and apply standardization + cleaning.
    Returns a DataFrame sorted by time (and match order where possible).
    """
    path = path_for_year(cfg, year)
    if not path.exists():
        raise FileNotFoundError(f"CSV for year {year} not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Normalize columns
    df = standardize_columns(df)

    # Check required columns early
    ensure_required_cols(df, REQUIRED_COLS, context=str(path))

    # Parse and create df["date"]
    df = parse_date(df)

    # Basic row filtering (incl. optional walkover removal)
    df = filter_basic_rows(df, remove_walkovers=cfg.remove_walkovers)

    # (Optional) singles-only filtering if dataset supports
    if cfg.keep_only_singles:
        df = filter_only_singles_if_possible(df)

    # Drop leakage columns AFTER walkover filtering (since walkover often uses score)
    if cfg.drop_leakage:
        df = drop_leakage_cols(df)

    # Sort deterministically
    df = sort_matches(df)

    return df


def read_years(cfg: DataIOConfig, years: Iterable[int]) -> pd.DataFrame:
    """Read multiple years and concatenate into a single cleaned DataFrame."""
    frames = [read_year(cfg, y) for y in years]
    df = pd.concat(frames, ignore_index=True)

    # After concat: enforce sort again (safe)
    df = sort_matches(df)
    return df


def read_range(cfg: DataIOConfig, start_year: int, end_year: int) -> pd.DataFrame:
    """Convenience: read years from start_year to end_year inclusive."""
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return read_years(cfg, range(start_year, end_year + 1))


# ----------------------------
# Helpers: standardize + validate
# ----------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and ensure basic types.
    - Lowercase column names, strip whitespace.
    - Keep tourney_date as string.
    - Force numeric types for ranks, ages, heights (handling 'NR' or strings).
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Make sure tourney_date is present before referencing
    if "tourney_date" in df.columns:
        df["tourney_date"] = df["tourney_date"].astype(str).str.strip()

    # CORREÇÃO CRÍTICA: Forçar conversão numérica e incluir match_num para ordenação correta
    cols_to_numeric = [
        "winner_rank", "loser_rank", 
        "winner_age", "loser_age", 
        "winner_ht", "loser_ht",
        "match_num"  # Essencial: garante que 1 < 2 < 10 (texto faria 1 < 10 < 2)
    ]
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def ensure_required_cols(df: pd.DataFrame, required: Iterable[str], context: str = "") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing required columns {missing}"
        if context:
            msg += f" in {context}"
        raise ValueError(msg)


# ----------------------------
# Helpers: parsing date
# ----------------------------

def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create df['date'] as datetime from 'tourney_date'.
    Handles:
    - 'YYYYMMDD' (most common)
    - other parseable date strings
    Drops rows where date cannot be parsed.
    """
    df = df.copy()
    raw = df["tourney_date"].astype(str).str.strip()

    # Try strict YYYYMMDD first (fast + reliable)
    dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")

    # Fallback for any weird formats that weren't parsed
    mask_bad = dt.isna()
    if mask_bad.any():
        dt2 = pd.to_datetime(raw[mask_bad], errors="coerce")
        dt.loc[mask_bad] = dt2

    df["date"] = dt
    df = df.dropna(subset=["date"])
    return df


# ----------------------------
# Helpers: row filtering
# ----------------------------

def filter_basic_rows(df: pd.DataFrame, *, remove_walkovers: bool) -> pd.DataFrame:
    """
    Remove invalid rows:
    - missing winner_id / loser_id
    - winner_id == loser_id
    - (optional) walkovers, if detectable via 'score'
    """
    df = df.copy()

    # Winner/Loser ids must exist
    df = df.dropna(subset=["winner_id", "loser_id"])

    # Remove impossible self-matches
    df = df[df["winner_id"] != df["loser_id"]]

    # Optional: remove walkovers if score exists
    if remove_walkovers and "score" in df.columns:
        score = df["score"].astype(str).str.upper()
        # Common markers: "W/O", "WO", "WALKOVER"
        is_wo = score.str.contains("W/O", na=False) | score.str.contains("WALKOVER", na=False)
        df = df[~is_wo]

    return df


def filter_only_singles_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to filter to singles only if dataset contains a relevant column.
    If no such column exists, returns df unchanged.
    """
    df = df.copy()

    # Different datasets name these differently; we try a few patterns.
    # If none exist, do nothing.
    candidates = [
        "match_type",     # e.g., "S" or "D"
        "event_type",
        "doubles",
        "is_doubles",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return df

    s = df[col].astype(str).str.strip().str.upper()
    # Heuristics:
    # - if values contain "S"/"D"
    # - or booleans for doubles
    if s.isin(["S", "SINGLES"]).any():
        return df[s.isin(["S", "SINGLES"])]
    if s.isin(["D", "DOUBLES"]).any():
        return df[~s.isin(["D", "DOUBLES"])]
    if s.isin(["TRUE", "FALSE"]).any():
        return df[s.isin(["FALSE"])]
    if s.isin(["0", "1"]).any():
        return df[s.isin(["0"])]

    # If it doesn't match any known schema, don't guess.
    return df


# ----------------------------
# Helpers: leakage control
# ----------------------------

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove post-match columns:
    - exact: score, minutes
    - prefixes: w_*, l_*
    """
    keep_cols: list[str] = []
    for c in df.columns:
        if c in LEAKAGE_EXACT:
            continue
        if c.startswith(LEAKAGE_PREFIXES):
            continue
        keep_cols.append(c)
    return df[keep_cols].copy()


# ----------------------------
# Helpers: sorting
# ----------------------------

def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort deterministically:
    - by date ascending
    - then by tourney_id (agrupa o torneio)
    - then by match_num (CRÍTICO: garante ordem correta R1 -> Final dentro do torneio)
    - else stable by winner_id/loser_id
    """
    df = df.copy()

    # Ordem de prioridade para evitar leakage intra-torneio
    sort_cols = ["date"]
    
    # Se tivermos tourney_id e match_num, eles são a verdade absoluta da ordem
    if "tourney_id" in df.columns:
        sort_cols.append("tourney_id")
    if "match_num" in df.columns:
        sort_cols.append("match_num")

    # Fallback stable cols (caso falte match_num, o que é raro em ATP oficial)
    for c in ("winner_id", "loser_id"):
        if c in df.columns and c not in sort_cols:
            sort_cols.append(c)

    df = df.sort_values(sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)
    return df


# ----------------------------
# Minimal manual test
# ----------------------------

if __name__ == "__main__":
    cfg = DataIOConfig(data_dir=Path("data"), drop_leakage=True, remove_walkovers=True)
    df = read_range(cfg, 2010, 2012)
    print("Loaded:", df.shape)
    print("Cols (sample):", list(df.columns)[:25])
    """print(df[["date", "winner_id", "loser_id"]].head())"""
    print("\nPrimeira linha organizada:")
    print(df.iloc[0].to_string())

