# ML/features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ✅ recomendado: imports relativos dentro do pacote ML
from .dataio import DataIOConfig, read_range


# ----------------------------
# Paths / cfg (pra rodar local)
# ----------------------------

# Paths / cfg
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# AJUSTE 1: Mude drop_leakage para False.
# Motivo: Queremos carregar as stats (aces, minutes) para calcular médias móveis se precisar,
# e só dropar essas colunas "proibidas" no final do pipeline, antes do treino.
cfg = DataIOConfig(data_dir=DATA_DIR, drop_leakage=False, remove_walkovers=True)



# ----------------------------
# Elo config + helpers
# ----------------------------

@dataclass(frozen=True)
class EloConfig:
    base: float = 1500.0
    k: float = 32.0
    k_new: float = 64.0
    provisional_games: int = 10
    add_prob: bool = True  # cria elo_prob_* também


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


# ----------------------------
# Features: add_elo_wl (ajustada)
# ----------------------------
def add_elo_wl(df_wl: pd.DataFrame, elo_cfg: EloConfig) -> pd.DataFrame:
    """
    Calcula Elo PRÉ-JOGO em df winner/loser (1 linha por match),
    atualizando o rating UMA vez por partida (sem duplicação).

    Requer colunas: date, winner_id, loser_id.
    """
    df = df_wl.copy()

    # Ordenação defensiva + determinística (inclui tourney_id se existir)
    sort_cols = ["date"]
    if "tourney_id" in df.columns:
        sort_cols.append("tourney_id")
    if "match_num" in df.columns:
        sort_cols.append("match_num")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    ratings: Dict[int, float] = {}
    games: Dict[int, int] = {}

    w_pre_list = []
    l_pre_list = []
    prob_w_list = []

    # itertuples é MUITO mais rápido que iterrows
    for row in df.itertuples(index=False):
        # ids podem vir como float (ex: 123.0), então força int via float()
        w = int(float(getattr(row, "winner_id")))
        l = int(float(getattr(row, "loser_id")))

        r_w = ratings.get(w, elo_cfg.base)
        r_l = ratings.get(l, elo_cfg.base)

        # Elo pré-jogo
        w_pre_list.append(r_w)
        l_pre_list.append(r_l)

        p_w = elo_expected(r_w, r_l)
        if elo_cfg.add_prob:
            prob_w_list.append(p_w)

        # K adaptativo (provisional)
        gw = games.get(w, 0)
        gl = games.get(l, 0)
        k_w = elo_cfg.k_new if gw < elo_cfg.provisional_games else elo_cfg.k
        k_l = elo_cfg.k_new if gl < elo_cfg.provisional_games else elo_cfg.k

        # Atualização
        r_w_new = r_w + k_w * (1.0 - p_w)
        r_l_new = r_l + k_l * (0.0 - (1.0 - p_w))

        ratings[w] = r_w_new
        ratings[l] = r_l_new
        games[w] = gw + 1
        games[l] = gl + 1

    df["winner_elo_pre"] = w_pre_list
    df["loser_elo_pre"] = l_pre_list
    df["elo_diff_wl"] = df["winner_elo_pre"] - df["loser_elo_pre"]
    if elo_cfg.add_prob:
        df["winner_elo_prob"] = prob_w_list

    return df



# ----------------------------
# Winner/Loser -> P1/P2 (duplica)
# ----------------------------

# ----------------------------
# Features: create_pairwise_data (ajustada)
# ----------------------------
def create_pairwise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dobra o dataset:
      - positivo: P1=winner, P2=loser, target=1
      - negativo: P1=loser, P2=winner, target=0

    ⚠️ Features temporais (Elo, forma recente, H2H etc.) devem ser calculadas ANTES.
    """
    df_pos = df.copy()
    cols_pos = {
        c: c.replace("winner_", "p1_").replace("loser_", "p2_")
        for c in df_pos.columns
        if (c.startswith("winner_") or c.startswith("loser_"))
    }
    df_pos = df_pos.rename(columns=cols_pos)
    df_pos["target"] = 1

    df_neg = df.copy()
    cols_neg = {
        c: c.replace("loser_", "p1_").replace("winner_", "p2_")
        for c in df_neg.columns
        if (c.startswith("winner_") or c.startswith("loser_"))
    }
    df_neg = df_neg.rename(columns=cols_neg)
    df_neg["target"] = 0

    df_pairwise = pd.concat([df_pos, df_neg], ignore_index=True)

    sort_cols = ["date"]
    if "tourney_id" in df_pairwise.columns:
        sort_cols.append("tourney_id")
    if "match_num" in df_pairwise.columns:
        sort_cols.append("match_num")

    df_pairwise = df_pairwise.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df_pairwise



# ----------------------------
# Features básicas no pairwise
# ----------------------------
# Em ML/features.py

# ----------------------------
# Features: add_basic_features_pairwise (ajustada)
# ----------------------------
def add_basic_features_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features simples (diferenças), flags de missing e encodings básicos.
    """
    out = df.copy()

    # --- Rank ---
    if "p1_rank" in out.columns and "p2_rank" in out.columns:
        out["p1_rank_missing"] = out["p1_rank"].isna().astype(int)
        out["p2_rank_missing"] = out["p2_rank"].isna().astype(int)
        out["p1_rank"] = out["p1_rank"].fillna(9999)
        out["p2_rank"] = out["p2_rank"].fillna(9999)
        out["rank_diff"] = out["p2_rank"] - out["p1_rank"]  # positivo => p1 melhor rankeado

    # --- Age ---
    if "p1_age" in out.columns and "p2_age" in out.columns:
        out["p1_age_missing"] = out["p1_age"].isna().astype(int)
        out["p2_age_missing"] = out["p2_age"].isna().astype(int)
        out["p1_age"] = out["p1_age"].fillna(28.0)
        out["p2_age"] = out["p2_age"].fillna(28.0)
        out["age_diff"] = out["p1_age"] - out["p2_age"]

    # --- Elo (diff + prob vectorizado) ---
    if "p1_elo_pre" in out.columns and "p2_elo_pre" in out.columns:
        ra = out["p1_elo_pre"].astype(float)
        rb = out["p2_elo_pre"].astype(float)
        out["elo_diff"] = ra - rb
        out["elo_prob_p1"] = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    # --- Surface ---
    if "surface" in out.columns:
        surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
        out["surface_code"] = out["surface"].map(surface_map).fillna(-1)

    # --- Hand + interação canhoto vs destro ---
    if "p1_hand" in out.columns:
        h1 = out["p1_hand"].fillna("U").astype(str).str.upper()
    else:
        h1 = pd.Series(["U"] * len(out), index=out.index)

    if "p2_hand" in out.columns:
        h2 = out["p2_hand"].fillna("U").astype(str).str.upper()
    else:
        h2 = pd.Series(["U"] * len(out), index=out.index)

    if "p1_hand" in out.columns:
        out["p1_hand_code"] = h1.map({"R": 0, "L": 1, "U": 2}).fillna(2)
    if "p2_hand" in out.columns:
        out["p2_hand_code"] = h2.map({"R": 0, "L": 1, "U": 2}).fillna(2)

    out["is_lefty_vs_righty"] = (((h1 == "L") & (h2 == "R")) | ((h1 == "R") & (h2 == "L"))).astype(int)

    # --- Round ---
    if "round" in out.columns:
        round_map = {
            "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
            "RR": 8,   # Round Robin
            "BR": 0,   # Bronze match (raro)
            "Q1": 0, "Q2": 1, "Q3": 2, "QS": 1  # Qualifiers (heurístico)
        }
        out["round_code"] = out["round"].astype(str).str.upper().map(round_map).fillna(1)

    return out



# ----------------------------
# Pipeline “oficial” de features
# ----------------------------

def build_pairwise_dataset(
    start_year: int,
    end_year: int,
    *,
    data_cfg: DataIOConfig,
    elo_cfg: Optional[EloConfig] = None,
) -> pd.DataFrame:
    """
    Carrega histórico completo -> Calcula Elo -> Gera Features -> Filtra Anos -> LIMPEZA FINAL BLINDADA.
    """
    # 1. Carregar SEMPRE desde 2010 para o Elo ter histórico real
    full_history_start = 2010
    load_start = min(start_year, full_history_start)

    # Carrega range expandido (sem dropar leakage ainda, configurado no dataio)
    df_wl = read_range(data_cfg, load_start, end_year)

    if elo_cfg is not None:
        df_wl = add_elo_wl(df_wl, elo_cfg)

    # Gera dataset A/B (pairwise)
    df_pw = create_pairwise_data(df_wl)
    df_pw = add_basic_features_pairwise(df_pw)

    # --- FILTRAGEM FINAL DE ANOS ---
    # Agora sim cortamos para o período que o usuário pediu
    if "date" in df_pw.columns:
        df_pw["year_temp"] = df_pw["date"].dt.year
        df_pw = df_pw[(df_pw["year_temp"] >= start_year) & (df_pw["year_temp"] <= end_year)]
        df_pw = df_pw.drop(columns=["year_temp"])

    # --- REMOÇÃO DE LEAKAGE E METADADOS (VERSÃO SEGURA) ---
    
    # 1. Metadados e IDs que causam Overfitting (o modelo decora quem é o jogador)
    cols_to_drop = {
        "score", "minutes", "match_num", "tourney_id", "tourney_date",
        "winner_id", "loser_id", 
        "p1_id", "p2_id", "p1_name", "p2_name" # CRÍTICO: Remover IDs transformados
    }

    # 2. Sufixos de estatísticas pós-jogo (Leakage)
    # Remove qualquer coisa que termine com _ace, _df, etc., seja w_, l_, p1_ ou p2_
    leakage_suffixes = [
        "_ace", "_df", "_svpt", "_1stin", "_1stwon", "_2ndwon", 
        "_svgms", "_bpsaved", "_bpfaced", "_rank_points"
    ]

    drop_list = list(cols_to_drop)
    
    # Varredura final nas colunas
    for c in df_pw.columns:
        # Se terminar com sufixo proibido
        for suffix in leakage_suffixes:
            if c.endswith(suffix):
                drop_list.append(c)
                break 
        
        # Se começar com prefixo original de stat (w_ ou l_) e não for ID
        if c.startswith(("w_", "l_")) and c not in ["w_id", "l_id"]:
             drop_list.append(c)

    # Executa o drop garantindo que não dê erro se a coluna já não existir
    df_pw = df_pw.drop(columns=list(set(drop_list)), errors="ignore")

    return df_pw



# ----------------------------
# Teste manual
# ----------------------------

if __name__ == "__main__":
    print(f"📂 DATA_DIR = {DATA_DIR}")

    elo_cfg = EloConfig(base=1500.0, k=32.0, k_new=64.0, provisional_games=10, add_prob=True)

    df_pw = build_pairwise_dataset(
        2015, 2023,
        data_cfg=cfg,
        elo_cfg=elo_cfg
    )

    print("✅ Pairwise pronto:", df_pw.shape)
    cols_show = [c for c in ["date","p1_id","p2_id","target","elo_diff","elo_prob_p1","rank_diff","age_diff","surface"] if c in df_pw.columns]
    print(df_pw[cols_show].head(10))
