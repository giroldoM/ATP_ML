# ML/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """
    Um fold de walk-forward por ano:
      - train_years: anos usados para treinar o modelo (supervisionado)
      - eval_years: anos usados para avaliar (normalmente 1 ano)
    """
    name: str
    train_years: Tuple[int, ...]
    eval_years: Tuple[int, ...]


@dataclass(frozen=True)
class SplitPlan:
    """
    Plano completo do projeto:
      - tuning_folds: walk-forward 2019-2023
      - final_val: treino 2010-2023, valida 2024
      - test: treino 2010-2024, testa 2025-2026 (blindado)
    """
    tuning_folds: Tuple[Fold, ...]
    final_val: Fold
    test: Fold


# ----------------------------
# Fold builders (por ano)
# ----------------------------

def walk_forward_folds(
    start_year: int,
    first_eval_year: int,
    last_eval_year: int,
    *,
    prefix: str = "wf",
) -> Tuple[Fold, ...]:
    """
    Ex:
      start_year=2010, first_eval_year=2019, last_eval_year=2023

      wf_2019: train 2010-2018, eval 2019
      wf_2020: train 2010-2019, eval 2020
      ...
      wf_2023: train 2010-2022, eval 2023
    """
    if last_eval_year < first_eval_year:
        raise ValueError("last_eval_year must be >= first_eval_year")
    if first_eval_year <= start_year:
        raise ValueError("first_eval_year must be > start_year")

    folds: List[Fold] = []
    for y in range(first_eval_year, last_eval_year + 1):
        train_years = tuple(range(start_year, y))
        eval_years = (y,)
        folds.append(Fold(name=f"{prefix}_{y}", train_years=train_years, eval_years=eval_years))
    return tuple(folds)


def make_split_plan(
    *,
    start_year: int = 2010,
    tuning_first: int = 2019,
    tuning_last: int = 2023,
    final_val_year: int = 2024,
    test_years: Tuple[int, int] = (2025, 2026),
) -> SplitPlan:
    """
    Cria o plano exatamente como vocês definiram.
    """
    tuning = walk_forward_folds(start_year, tuning_first, tuning_last, prefix="wf")

    final_val = Fold(
        name=f"val_{final_val_year}",
        train_years=tuple(range(start_year, final_val_year)),
        eval_years=(final_val_year,),
    )

    test_start, test_end = test_years
    if test_end < test_start:
        raise ValueError("test_years must be (start<=end)")

    test = Fold(
        name=f"test_{test_start}_{test_end}",
        train_years=tuple(range(start_year, test_start)),    # 2010-2024
        eval_years=tuple(range(test_start, test_end + 1)),   # 2025-2026
    )

    return SplitPlan(tuning_folds=tuning, final_val=final_val, test=test)


# ----------------------------
# Apply splits to a DataFrame
# ----------------------------

def ensure_year_column(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """
    Garante uma coluna 'year' baseada em df['date'].
    (Serve tanto pra WL quanto pra pairwise.)
    """
    if year_col in df.columns:
        return df
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' to derive year.")
    out = df.copy()
    out[year_col] = pd.to_datetime(out["date"]).dt.year.astype(int)
    return out


def slice_years(df: pd.DataFrame, years: Iterable[int], *, year_col: str = "year") -> pd.DataFrame:
    """
    Retorna apenas as linhas cujo ano está em 'years'.
    """
    years_set = set(int(y) for y in years)
    out = ensure_year_column(df, year_col=year_col)
    return out[out[year_col].isin(years_set)].copy()


def split_df_by_fold(
    df_with_features: pd.DataFrame,
    fold: Fold,
    *,
    year_col: str = "year",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (train_df, eval_df) dado um DF já com features calculadas.

    ⚠️ IMPORTANTE (Elo):
      df_with_features deve ter sido gerado em um range contínuo que inclua
      tanto train quanto eval (ex.: 2010-2024),
      para que o Elo pré-jogo em eval não resete.
    """
    train_df = slice_years(df_with_features, fold.train_years, year_col=year_col)
    eval_df = slice_years(df_with_features, fold.eval_years, year_col=year_col)
    return train_df, eval_df


# ----------------------------
# Minimal self-test
# ----------------------------

if __name__ == "__main__":
    plan = make_split_plan()
    print("TUNING folds:")
    for f in plan.tuning_folds:
        print(f" - {f.name}: train {f.train_years[0]}..{f.train_years[-1]} | eval {f.eval_years}")

    print("\nFinal val:", plan.final_val)
    print("Test:", plan.test)
