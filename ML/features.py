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
# Em ML/features.py

def add_basic_features_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features simples (diferenças), codificações e interações (ex: canhoto vs destro).
    """
    out = df.copy()

    # --- Rank / Age ---
    if "p1_rank" in out.columns and "p2_rank" in out.columns:
        out["p1_rank"] = out["p1_rank"].fillna(9999)
        out["p2_rank"] = out["p2_rank"].fillna(9999)
        out["rank_diff"] = out["p2_rank"] - out["p1_rank"]  # positivo => p1 melhor rankeado

    if "p1_age" in out.columns and "p2_age" in out.columns:
        out["p1_age_missing"] = out["p1_age"].isna().astype(int)
        out["p2_age_missing"] = out["p2_age"].isna().astype(int)
        out["p1_age"] = out["p1_age"].fillna(28.0)
        out["p2_age"] = out["p2_age"].fillna(28.0)
        out["age_diff"] = out["p1_age"] - out["p2_age"]

    # --- Elo ---
    if "p1_elo_pre" in out.columns and "p2_elo_pre" in out.columns:
        out["elo_diff"] = out["p1_elo_pre"] - out["p2_elo_pre"]
        out["elo_prob_p1"] = out["p1_elo_pre"].combine(
            out["p2_elo_pre"],
            lambda ra, rb: elo_expected(float(ra), float(rb))
        )

    # --- Surface Code ---
    if "surface" in out.columns:
        # Hard=0, Clay=1, Grass=2, Carpet=3 (ou use OneHot depois)
        surface_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
        out["surface_code"] = out["surface"].map(surface_map).fillna(-1)

    # --- Hand & Matchup (NOVO) ---
    # Normaliza para R (Right), L (Left), U (Unknown)
    h1 = out["p1_hand"].fillna("U") if "p1_hand" in out.columns else pd.Series(["U"] * len(out))
    h2 = out["p2_hand"].fillna("U") if "p2_hand" in out.columns else pd.Series(["U"] * len(out))

    if "p1_hand" in out.columns:
        out["p1_hand_code"] = h1.map({"R": 0, "L": 1, "U": 2}).fillna(2)
    if "p2_hand" in out.columns:
        out["p2_hand_code"] = h2.map({"R": 0, "L": 1, "U": 2}).fillna(2)

    # Feature de interação: Jogo é Canhoto vs Destro? (1=Sim, 0=Não)
    # Isso costuma ser relevante taticamente.
    # (h1='L' e h2='R') OU (h1='R' e h2='L')
    is_l_vs_r = ((h1 == "L") & (h2 == "R")) | ((h1 == "R") & (h2 == "L"))
    out["is_lefty_vs_righty"] = is_l_vs_r.astype(int)

    # --- Round Code ---
    if "round" in out.columns:
        round_map = {
            "F": 7, "SF": 6, "QF": 5,
            "R16": 4, "R32": 3, "R64": 2, "R128": 1,
            "RR": 8,  # CORREÇÃO: Round Robin (Finals) vale muito, não 0
            "BR": 0,  # Bronze match (raro)
            "QF": 5, "QS": 1, "Q1": 0, "Q2": 1  # Qualifiers
        }
        # Mapeia e preenche não encontrados com 1 (assumindo round inicial padrão se falhar)
        out["round_code"] = out["round"].map(round_map).fillna(1)

    return out


# ----------------------------
# Pipeline “oficial” de features
# ----------------------------

# Em ML/features.py

def build_pairwise_dataset(
    start_year: int,
    end_year: int,
    *,
    data_cfg: DataIOConfig,
    elo_cfg: Optional[EloConfig] = None,
) -> pd.DataFrame:
    """
    Carrega WL -> (opcional Elo) -> duplica -> features básicas -> LIMPEZA FINAL.
    """
    # DICA DE OURO: Sempre carregue desde 2010 (ou antes) para o Elo "pegar tração".
    # Se carregar só 2023-2024, o Elo começa frio (1500) em 2023, o que é ruim.
    # Carregamos tudo, calculamos features, e DEPOIS filtramos os anos pedidos.
    full_start = 2010  # ou o ano min do seu dataset
    df_wl = read_range(data_cfg, full_start, end_year)

    if elo_cfg is not None:
        df_wl = add_elo_wl(df_wl, elo_cfg)
    
    # Aqui entrariam outras funções de features em W/L (ex: Rolling Stats de Aces/BreakPoints)
    # ...

    df_pw = create_pairwise_data(df_wl)
    df_pw = add_basic_features_pairwise(df_pw)

    # --- FILTRAGEM FINAL DE ANOS ---
    # Agora sim cortamos para o período que o usuário pediu (ex: treino 2015-2023)
    df_pw["year"] = df_pw["date"].dt.year
    df_pw = df_pw[(df_pw["year"] >= start_year) & (df_pw["year"] <= end_year)].copy()

    # --- REMOÇÃO DE LEAKAGE (FINAL) ---
    # Como desligamos o drop_leakage=False na config para poder calcular features,
    # agora precisamos limpar a sujeira (stats pós-jogo) para não vazar no treino.
    # Lista de colunas perigosas típicas do ATP (adapte se tiver mais):
    leakage_cols = [
        "score", "minutes", "match_num", "winner_id", "loser_id",
        "w_ace", "w_df", "w_svpt", "w_1stin", "w_1stwon", "w_2ndwon", "w_svgms", "w_bpsaved", "w_bpfaced",
        "l_ace", "l_df", "l_svpt", "l_1stin", "l_1stwon", "l_2ndwon", "l_svgms", "l_bpsaved", "l_bpfaced",
        "winner_rank_points", "loser_rank_points" # pontos do ranking as vezes vazam futuro dependendo da fonte
    ]
    # Remove também as versões "p1_" e "p2_" dessas stats que o pairwise criou
    cols_to_drop = []
    for c in df_pw.columns:
        # Se for coluna de leakage crua ou transformada (ex: p1_ace)
        base_name = c.replace("p1_", "").replace("p2_", "").replace("winner_", "").replace("loser_", "")
        if base_name in leakage_cols or c in leakage_cols:
            cols_to_drop.append(c)
    
    df_pw = df_pw.drop(columns=cols_to_drop, errors="ignore")

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
