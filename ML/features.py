# ML/features.py
"""
Módulo de Engenharia de Features.
Responsável pelo cálculo de métricas históricas (Elo Rating) e preparação do dataset
no formato Pairwise (P1 vs P2) para modelagem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Import relativo para manter o pacote coeso
from .dataio import DataIOConfig, read_range, drop_leakage_cols

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EloConfig:
    """Parâmetros para o algoritmo de Elo Rating."""
    base: float = 1500.0
    k: float = 32.0          # Fator K padrão para jogadores estabelecidos
    k_new: float = 64.0      # Fator K acelerado para novatos
    provisional_games: int = 10 # Número de jogos considerados "fase de calibração"
    add_prob: bool = True    # Se True, adiciona a probabilidade implícita do Elo

# -----------------------------------------------------------------------------
# Elo Calculation Engine
# -----------------------------------------------------------------------------

def _calculate_elo_probability(r_a: float, r_b: float) -> float:
    """Calcula a probabilidade esperada de A vencer B dado seus ratings."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def add_elo_wl(df_wl: pd.DataFrame, elo_cfg: EloConfig) -> pd.DataFrame:
    """
    Calcula e anexa o Elo Rating PRÉ-JOGO (Global e por Superfície) ao DataFrame.
    
    O algoritmo itera cronologicamente partida a partida.
    IMPORTANTE: Os valores anexados são sempre o rating *antes* da partida acontecer,
    evitando Data Leakage.
    """
    df = df_wl.copy()
    
    # Dicionários de estado (Memória do sistema Elo)
    # Global
    ratings_global: Dict[int, float] = {}
    games_global: Dict[int, int] = {}
    
    # Por Superfície: {'Hard': {id: rating}, 'Clay': ...}
    surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    ratings_surface: Dict[str, Dict[int, float]] = {s: {} for s in surfaces}
    games_surface: Dict[str, Dict[int, int]] = {s: {} for s in surfaces}

    # Buffers para armazenar colunas (mais rápido que alocar no DF a cada loop)
    cols_data = {
        "winner_elo_pre": [], "loser_elo_pre": [],
        "winner_elo_surface": [], "loser_elo_surface": []
    }

    # Iteração linear (necessária para Elo pois depende do estado anterior)
    for row in df.itertuples(index=False):
        w_id = int(getattr(row, "winner_id"))
        l_id = int(getattr(row, "loser_id"))
        
        # Normalização da superfície
        surf_raw = getattr(row, "surface", "Hard")
        surf = str(surf_raw).strip().title() if isinstance(surf_raw, str) else "Hard"
        if surf not in ratings_surface:
            surf = "Hard" # Fallback conservador

        # --- 1. Elo Global ---
        r_w_g = ratings_global.get(w_id, elo_cfg.base)
        r_l_g = ratings_global.get(l_id, elo_cfg.base)
        
        cols_data["winner_elo_pre"].append(r_w_g)
        cols_data["loser_elo_pre"].append(r_l_g)

        # Atualização Global (Post-match)
        prob_w_g = _calculate_elo_probability(r_w_g, r_l_g)
        
        # K dinâmico baseado na experiência do jogador
        gw_g = games_global.get(w_id, 0)
        gl_g = games_global.get(l_id, 0)
        k_w = elo_cfg.k_new if gw_g < elo_cfg.provisional_games else elo_cfg.k
        k_l = elo_cfg.k_new if gl_g < elo_cfg.provisional_games else elo_cfg.k
        
        ratings_global[w_id] = r_w_g + k_w * (1.0 - prob_w_g)
        ratings_global[l_id] = r_l_g + k_l * (0.0 - (1.0 - prob_w_g))
        games_global[w_id] = gw_g + 1
        games_global[l_id] = gl_g + 1

        # --- 2. Elo por Superfície ---
        r_dict = ratings_surface[surf]
        g_dict = games_surface[surf]
        
        r_w_s = r_dict.get(w_id, elo_cfg.base)
        r_l_s = r_dict.get(l_id, elo_cfg.base)
        
        cols_data["winner_elo_surface"].append(r_w_s)
        cols_data["loser_elo_surface"].append(r_l_s)

        # Atualização Surface (Post-match)
        prob_w_s = _calculate_elo_probability(r_w_s, r_l_s)
        gw_s = g_dict.get(w_id, 0)
        gl_s = g_dict.get(l_id, 0)
        
        k_w_s = elo_cfg.k_new if gw_s < elo_cfg.provisional_games else elo_cfg.k
        k_l_s = elo_cfg.k_new if gl_s < elo_cfg.provisional_games else elo_cfg.k
        
        r_dict[w_id] = r_w_s + k_w_s * (1.0 - prob_w_s)
        r_dict[l_id] = r_l_s + k_l_s * (0.0 - (1.0 - prob_w_s))
        g_dict[w_id] = gw_s + 1
        g_dict[l_id] = gl_s + 1

    # Atribuição em lote ao DataFrame
    for col_name, values in cols_data.items():
        df[col_name] = values

    return df

# -----------------------------------------------------------------------------
# Dataset Transformation (W/L -> Pairwise)
# -----------------------------------------------------------------------------

def create_pairwise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o formato Winner/Loser (W/L) em Player 1 / Player 2 (P1/P2).
    
    Duplica os dados para garantir simetria:
    - Cenário A: P1 = Winner, P2 = Loser, Target = 1
    - Cenário B: P1 = Loser, P2 = Winner, Target = 0
    """
    # 1. Criação do lado Positivo (Winner é P1)
    df_pos = df.copy()
    cols_map_pos = {
        c: c.replace("winner_", "p1_").replace("loser_", "p2_")
        for c in df_pos.columns if c.startswith(("winner_", "loser_"))
    }
    df_pos = df_pos.rename(columns=cols_map_pos)
    df_pos["target"] = 1

    # 2. Criação do lado Negativo (Loser é P1)
    df_neg = df.copy()
    cols_map_neg = {
        c: c.replace("loser_", "p1_").replace("winner_", "p2_")
        for c in df_neg.columns if c.startswith(("winner_", "loser_"))
    }
    df_neg = df_neg.rename(columns=cols_map_neg)
    df_neg["target"] = 0

    # Concatena e reordena
    df_pairwise = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # Ordenação final
    sort_cols = ["date"]
    if "tourney_id" in df_pairwise.columns: sort_cols.append("tourney_id")
    if "match_num" in df_pairwise.columns: sort_cols.append("match_num")
    
    return df_pairwise.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

def add_basic_features_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Gera features diferenciais (P1 - P2) e tratamentos de nulos."""
    out = df.copy()

    # Helpers para imputação
    def fill_and_diff(col_base, fill_val, suffix=""):
        p1_col, p2_col = f"p1_{col_base}", f"p2_{col_base}"
        if p1_col in out.columns and p2_col in out.columns:
            out[f"{p1_col}_missing"] = out[p1_col].isna().astype(int)
            out[f"{p2_col}_missing"] = out[p2_col].isna().astype(int)
            
            out[p1_col] = out[p1_col].fillna(fill_val)
            out[p2_col] = out[p2_col].fillna(fill_val)
            
            diff_col = f"{col_base}_diff{suffix}"
            out[diff_col] = out[p1_col] - out[p2_col]
            
            # Para Rank, menor é melhor. Se P1 < P2, P1 é favorito.
            # Delta normal = P1 - P2. Se for negativo, P1 é favorito.
            if col_base == "rank":
                out[diff_col] = out[p2_col] - out[p1_col] # Inverte para: Positivo = P1 tem rank melhor
            else:
                out[diff_col] = out[p1_col] - out[p2_col]

    fill_and_diff("rank", 9999)
    fill_and_diff("age", 28.0)

    # Elo Diff e Probabilidades
    if "p1_elo_pre" in out.columns and "p2_elo_pre" in out.columns:
        ra, rb = out["p1_elo_pre"], out["p2_elo_pre"]
        out["elo_diff"] = ra - rb
        out["elo_prob_p1"] = _calculate_elo_probability(ra, rb)

    if "p1_elo_surface" in out.columns and "p2_elo_surface" in out.columns:
        ra_s, rb_s = out["p1_elo_surface"], out["p2_elo_surface"]
        out["elo_diff_surface"] = ra_s - rb_s
        out["elo_prob_surface_p1"] = _calculate_elo_probability(ra_s, rb_s)

    # Encodings Categóricos Simples
    if "surface" in out.columns:
        surf_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
        out["surface_code"] = out["surface"].map(surf_map).fillna(-1)

    # Hand (Canhoto vs Destro)
    hands = {"R": 0, "L": 1, "U": 2}
    for p in ["p1", "p2"]:
        if f"{p}_hand" in out.columns:
            out[f"{p}_hand_code"] = out[f"{p}_hand"].fillna("U").str.upper().map(hands).fillna(2)
            
    # Interação Específica (Lefty vs Righty)
    if "p1_hand_code" in out.columns and "p2_hand_code" in out.columns:
        h1, h2 = out["p1_hand_code"], out["p2_hand_code"]
        # 0=Right, 1=Left.
        out["is_lefty_vs_righty"] = ((h1 == 0) & (h2 == 1)) | ((h1 == 1) & (h2 == 0))
        out["is_lefty_vs_righty"] = out["is_lefty_vs_righty"].astype(int)

    return out

# -----------------------------------------------------------------------------
# Main Pipeline Builder
# -----------------------------------------------------------------------------

def build_pairwise_dataset(
    start_year: int,
    end_year: int,
    *,
    data_cfg: DataIOConfig,
    elo_cfg: Optional[EloConfig] = None,
) -> pd.DataFrame:
    """
    Pipeline completo: 
    Carrega -> Calcula Elo Histórico -> Remove Vazamento -> Transforma Pairwise -> Limpa.
    """
    # 1. Carregamento Histórico (Elo precisa de passado para convergir)
    # Fixamos 2010 como início do "Big Bang" dos dados para ter histórico
    load_start_year = 2010 
    
    # Forçamos drop_leakage=False aqui para ter acesso aos metadados durante cálculo de features
    # (ex: estatísticas passadas). O drop real acontece depois.
    temp_cfg = dataclass_replace(data_cfg, drop_leakage=False)
    
    df_wl = read_range(temp_cfg, load_start_year, end_year)

    # 2. Cálculo de Features Temporais (Elo)
    if elo_cfg:
        df_wl = add_elo_wl(df_wl, elo_cfg)

    # 3. Remoção Segura de Leakage
    # Agora removemos colunas como 'score', 'w_ace', etc, pois o jogo já "aconteceu"
    df_wl = drop_leakage_cols(df_wl)

    # 4. Transformação Pairwise
    df_pw = create_pairwise_data(df_wl)
    df_pw = add_basic_features_pairwise(df_pw)

    # 5. Filtragem Final de Data
    # Cortamos apenas os anos solicitados pelo usuário para treino/teste
    if "date" in df_pw.columns:
        year_col = df_pw["date"].dt.year
        df_pw = df_pw[(year_col >= start_year) & (year_col <= end_year)]

    # 6. Limpeza de Metadados
    # Removemos IDs e colunas de texto que não servem para o modelo
    cols_drop = [
        "match_num", "tourney_id", "tourney_date", 
        "winner_id", "loser_id", "p1_id", "p2_id", 
        "p1_name", "p2_name"
    ]
    df_pw = df_pw.drop(columns=cols_drop, errors="ignore")

    return df_pw

def dataclass_replace(obj, **kwargs):
    """Helper para substituir valores em dataclasses frozen (Python < 3.10 clean fix)."""
    from dataclasses import replace
    return replace(obj, **kwargs)