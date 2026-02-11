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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Por padrão: sem leakage no DF (walkover é filtrado antes no dataio)
cfg = DataIOConfig(data_dir=DATA_DIR, drop_leakage=True, remove_walkovers=True)


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


def add_elo_wl(df_wl: pd.DataFrame, elo_cfg: EloConfig) -> pd.DataFrame:
    """
    Calcula Elo PRÉ-JOGO em df winner/loser (1 linha por match),
    atualizando o rating UMA vez por partida (sem duplicação).

    Requer colunas: date, winner_id, loser_id.
    (o dataio já garante date e ids, e já ordena; mesmo assim, reforçamos sort por segurança)
    """
    df = df_wl.copy()

    # Ordenação defensiva (dataio já faz isso)
    sort_cols = ["date"]
    if "match_num" in df.columns:
        sort_cols.append("match_num")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    ratings: Dict[int, float] = {}
    games: Dict[int, int] = {}

    w_pre_list = []
    l_pre_list = []
    prob_w_list = []

    for _, row in df.iterrows():
        w = int(row["winner_id"])
        l = int(row["loser_id"])

        r_w = ratings.get(w, elo_cfg.base)
        r_l = ratings.get(l, elo_cfg.base)

        # Elo pré-jogo
        w_pre_list.append(r_w)
        l_pre_list.append(r_l)

        # Probabilidade (opcional)
        p_w = elo_expected(r_w, r_l)
        if elo_cfg.add_prob:
            prob_w_list.append(p_w)

        # K adaptativo para novatos (provisional)
        gw = games.get(w, 0)
        gl = games.get(l, 0)
        k_w = elo_cfg.k_new if gw < elo_cfg.provisional_games else elo_cfg.k
        k_l = elo_cfg.k_new if gl < elo_cfg.provisional_games else elo_cfg.k

        # Atualiza (winner score=1, loser score=0)
        # Se k_w != k_l, atualiza separadamente (normal)
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

def create_pairwise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dobra o dataset:
      - positivo: P1=winner, P2=loser, target=1
      - negativo: P1=loser, P2=winner, target=0

    ⚠️ IMPORTANTE:
    - Features temporais (Elo, forma recente, H2H etc.) devem ser calculadas ANTES.
    """
    df_pos = df.copy()
    cols_pos = {
        c: c.replace("winner_", "p1_").replace("loser_", "p2_")
        for c in df_pos.columns
        if ("winner_" in c) or ("loser_" in c)
    }
    df_pos = df_pos.rename(columns=cols_pos)
    df_pos["target"] = 1

    df_neg = df.copy()
    cols_neg = {
        c: c.replace("loser_", "p1_").replace("winner_", "p2_")
        for c in df_neg.columns
        if ("winner_" in c) or ("loser_" in c)
    }
    df_neg = df_neg.rename(columns=cols_neg)
    df_neg["target"] = 0

    df_pairwise = pd.concat([df_pos, df_neg], ignore_index=True)

    # Ordenação segura (date sempre existe)
    sort_cols = ["date"]
    if "match_num" in df_pairwise.columns:
        sort_cols.append("match_num")
    df_pairwise = df_pairwise.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return df_pairwise


# ----------------------------
# Features básicas no pairwise
# ----------------------------

def add_basic_features_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features simples (diferenças) e algumas codificações.
    Nota: idealmente, categóricas (surface, hand) ficam cruas e são one-hot no training.
    """
    out = df.copy()

    # --- Rank / Age (robusto a colunas faltantes) ---
    if "p1_rank" in out.columns and "p2_rank" in out.columns:
        out["p1_rank"] = out["p1_rank"].fillna(9999)
        out["p2_rank"] = out["p2_rank"].fillna(9999)
        out["rank_diff"] = out["p2_rank"] - out["p1_rank"]  # positivo => p1 melhor

    if "p1_age" in out.columns and "p2_age" in out.columns:
        # imputação simples + flags (melhor do que usar mean sem flag)
        out["p1_age_missing"] = out["p1_age"].isna().astype(int)
        out["p2_age_missing"] = out["p2_age"].isna().astype(int)
        out["p1_age"] = out["p1_age"].fillna(28.0)
        out["p2_age"] = out["p2_age"].fillna(28.0)
        out["age_diff"] = out["p1_age"] - out["p2_age"]

    # --- Elo (vem do WL renomeado) ---
    if "p1_elo_pre" in out.columns and "p2_elo_pre" in out.columns:
        out["elo_diff"] = out["p1_elo_pre"] - out["p2_elo_pre"]
        out["elo_prob_p1"] = out["p1_elo_pre"].combine(
            out["p2_elo_pre"],
            lambda ra, rb: elo_expected(float(ra), float(rb))
        )

    # --- Surface / Hand / Round (opcional codificar aqui) ---
    # Eu deixaria cru e one-hot no training, mas se vocês quiserem manter código numérico:
    if "surface" in out.columns:
        surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
        out["surface_code"] = out["surface"].map(surface_map).fillna(-1)

    if "p1_hand" in out.columns:
        out["p1_hand"] = out["p1_hand"].fillna("U")
        out["p1_hand_code"] = out["p1_hand"].map({"R": 0, "L": 1, "U": 2}).fillna(2)

    if "p2_hand" in out.columns:
        out["p2_hand"] = out["p2_hand"].fillna("U")
        out["p2_hand_code"] = out["p2_hand"].map({"R": 0, "L": 1, "U": 2}).fillna(2)

    if "round" in out.columns:
        round_map = {
            "F": 7, "SF": 6, "QF": 5,
            "R16": 4, "R32": 3, "R64": 2, "R128": 1,
            "RR": 0, "BR": 6,
            # qualifiers comuns
            "QF": 5, "QS": 1, "Q1": 0, "Q2": 1
        }
        out["round_code"] = out["round"].map(round_map).fillna(0)

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
    Carrega WL -> (opcional Elo) -> duplica -> features básicas.
    """
    df_wl = read_range(data_cfg, start_year, end_year)

    if elo_cfg is not None:
        df_wl = add_elo_wl(df_wl, elo_cfg)

    df_pw = create_pairwise_data(df_wl)
    df_pw = add_basic_features_pairwise(df_pw)
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
