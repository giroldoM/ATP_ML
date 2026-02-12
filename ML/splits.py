# ML/splits.py
"""
Definição da estratégia de validação cruzada (Cross-Validation) e divisão de dados.
Utiliza estratégia Walk-Forward para respeitar a temporalidade dos esportes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Iterable

import pandas as pd

@dataclass(frozen=True)
class Fold:
    """Representa um conjunto de Treino/Validação."""
    name: str
    train_years: Tuple[int, ...]
    eval_years: Tuple[int, ...]

@dataclass(frozen=True)
class SplitPlan:
    """Plano mestre contendo todas as etapas de validação do projeto."""
    tuning_folds: Tuple[Fold, ...] # Folds para ajuste de hiperparâmetros
    final_val: Fold                # Validação final antes do teste cego
    test: Fold                     # Teste cego (futuro)

def make_split_plan(
    start_year: int = 2010,
    tuning_first: int = 2019,
    tuning_last: int = 2023,
    final_val_year: int = 2024,
    test_years: Tuple[int, int] = (2025, 2026),
) -> SplitPlan:
    """Gera o plano de cortes temporais."""
    
    # 1. Tuning (Walk-Forward)
    tuning_folds = []
    for year in range(tuning_first, tuning_last + 1):
        fold = Fold(
            name=f"wf_{year}",
            train_years=tuple(range(start_year, year)),
            eval_years=(year,)
        )
        tuning_folds.append(fold)

    # 2. Final Validation
    final_val = Fold(
        name=f"val_{final_val_year}",
        train_years=tuple(range(start_year, final_val_year)),
        eval_years=(final_val_year,)
    )

    # 3. Blind Test
    test = Fold(
        name="test_blind",
        train_years=tuple(range(start_year, test_years[0])),
        eval_years=tuple(range(test_years[0], test_years[1] + 1))
    )

    return SplitPlan(
        tuning_folds=tuple(tuning_folds),
        final_val=final_val,
        test=test
    )

def split_df_by_fold(
    df: pd.DataFrame, 
    fold: Fold, 
    year_col: str = "year"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fatia o DataFrame principal em Treino e Validação baseado no Fold.
    Garante que a coluna de ano exista.
    """
    df_processed = _ensure_year_column(df, year_col)
    
    train_mask = df_processed[year_col].isin(fold.train_years)
    eval_mask = df_processed[year_col].isin(fold.eval_years)
    
    return df_processed[train_mask].copy(), df_processed[eval_mask].copy()

def _ensure_year_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Helper para garantir que a coluna de ano derivada da data exista."""
    if col_name in df.columns:
        return df
    
    if "date" not in df.columns:
        raise ValueError("DataFrame precisa ter a coluna 'date' para derivar o ano.")
        
    out = df.copy()
    out[col_name] = pd.to_datetime(out["date"]).dt.year
    return out