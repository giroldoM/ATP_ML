# ML/dataio.py
"""
Módulo responsável pelo carregamento, padronização e limpeza inicial dos dados da ATP.
Garante que os DataFrames retornados tenham esquema consistente e tipos corretos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DataIOConfig:
    """Configuração para leitura e processamento dos arquivos CSV."""
    data_dir: Path
    filename_template: str = "atp_matches_{year}.csv"
    drop_leakage: bool = True
    remove_walkovers: bool = True
    keep_only_singles: bool = False

# Colunas essenciais para o funcionamento do pipeline
REQUIRED_COLS = ("tourney_date", "winner_id", "loser_id")

# Colunas que vazam informação do futuro (pós-jogo)
LEAKAGE_EXACT = {"score", "minutes"}
LEAKAGE_PREFIXES = ("w_", "l_")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def read_range(cfg: DataIOConfig, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Lê e concatena dados de partidas de um intervalo de anos (inclusive).
    
    Args:
        cfg: Configuração de diretórios e filtros.
        start_year: Ano inicial.
        end_year: Ano final.
        
    Returns:
        pd.DataFrame: DataFrame único contendo todo o histórico limpo e ordenado.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) deve ser maior ou igual a start_year ({start_year})")
    
    years = range(start_year, end_year + 1)
    return read_years(cfg, years)

def read_years(cfg: DataIOConfig, years: Iterable[int]) -> pd.DataFrame:
    """Lê múltiplos anos e concatena os resultados."""
    frames = [read_year(cfg, y) for y in years]
    if not frames:
        raise ValueError("Nenhum dado foi carregado. Verifique a lista de anos.")
        
    df = pd.concat(frames, ignore_index=True)
    return _sort_matches(df)

def read_year(cfg: DataIOConfig, year: int) -> pd.DataFrame:
    """
    Carrega o arquivo CSV de um ano específico, aplica padronização e filtros básicos.
    """
    file_path = cfg.data_dir / cfg.filename_template.format(year=year)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {file_path}")

    # Low_memory=False previne avisos de dtypes mistos em arquivos grandes
    df = pd.read_csv(file_path, low_memory=False)

    # Pipeline de limpeza
    df = _standardize_columns(df)
    _ensure_required_cols(df, REQUIRED_COLS, context=str(file_path))
    df = _parse_date(df)
    df = _filter_basic_rows(df, remove_walkovers=cfg.remove_walkovers)

    if cfg.keep_only_singles:
        df = _filter_only_singles(df)

    # Remoção de leakage (estatísticas pós-jogo)
    if cfg.drop_leakage:
        df = drop_leakage_cols(df)

    return _sort_matches(df)

def drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas que contêm estatísticas geradas após o término da partida."""
    # Mantém apenas colunas que NÃO estão na lista de exclusão
    keep_cols = [
        c for c in df.columns 
        if c not in LEAKAGE_EXACT and not c.startswith(LEAKAGE_PREFIXES)
    ]
    return df[keep_cols].copy()

# -----------------------------------------------------------------------------
# Internal Helpers (Private)
# -----------------------------------------------------------------------------

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas (lowercase) e força tipos numéricos onde necessário."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "tourney_date" in df.columns:
        df["tourney_date"] = df["tourney_date"].astype(str).str.strip()

    cols_numeric = [
        "winner_id", "loser_id", "winner_rank", "loser_rank",
        "winner_age", "loser_age", "winner_ht", "loser_ht", "match_num"
    ]
    
    for col in cols_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df

def _ensure_required_cols(df: pd.DataFrame, required: Iterable[str], context: str = "") -> None:
    """Valida se as colunas essenciais existem no DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Colunas obrigatórias ausentes {missing}"
        if context:
            msg += f" em {context}"
        raise ValueError(msg)

def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    """Converte 'tourney_date' para datetime e cria a coluna 'date'."""
    df = df.copy()
    raw_dates = df["tourney_date"].astype(str).str.strip()

    # Tenta formato padrão YYYYMMDD (mais rápido e comum)
    dates = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")

    # Fallback para outros formatos
    if dates.isna().any():
        mask_na = dates.isna()
        dates.loc[mask_na] = pd.to_datetime(raw_dates[mask_na], errors="coerce")

    df["date"] = dates
    # Remove partidas onde a data não pôde ser determinada
    return df.dropna(subset=["date"])

def _filter_basic_rows(df: pd.DataFrame, remove_walkovers: bool) -> pd.DataFrame:
    """Remove linhas inválidas (IDs nulos, IDs iguais) e W.O."""
    df = df.dropna(subset=["winner_id", "loser_id"])
    df = df[df["winner_id"] != df["loser_id"]]

    if remove_walkovers and "score" in df.columns:
        score = df["score"].astype(str).str.upper()
        # Regex simplificado para detectar W/O
        is_wo = score.str.contains(r"\b(W/O|WO|WALKOVER)\b", regex=True, na=False)
        df = df[~is_wo]

    return df

def _filter_only_singles(df: pd.DataFrame) -> pd.DataFrame:
    """Tenta filtrar apenas jogos de simples baseado em heurísticas de colunas comuns."""
    # Lista de possíveis nomes de colunas que indicam o tipo de jogo
    candidates = ["match_type", "event_type", "doubles", "is_doubles"]
    col = next((c for c in candidates if c in df.columns), None)
    
    if not col:
        return df

    s = df[col].astype(str).str.strip().str.upper()
    
    if s.isin(["S", "SINGLES"]).any():
        return df[s.isin(["S", "SINGLES"])]
    if s.isin(["D", "DOUBLES"]).any():
        return df[~s.isin(["D", "DOUBLES"])]
    if s.isin(["0", "1"]).any(): # Assumindo 0 = Singles, 1 = Doubles
        return df[s == "0"]
        
    return df

def _sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordenação determinística crítica para evitar Data Leakage em séries temporais.
    Ordem: Data -> Torneio -> Número da Partida.
    """
    sort_cols = ["date"]
    if "tourney_id" in df.columns: sort_cols.append("tourney_id")
    if "match_num" in df.columns: sort_cols.append("match_num")
    
    # Fallback para IDs se match_num não existir (raro)
    for c in ("winner_id", "loser_id"):
        if c in df.columns and c not in sort_cols:
            sort_cols.append(c)

    return df.sort_values(sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)