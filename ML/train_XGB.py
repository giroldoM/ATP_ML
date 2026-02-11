# ML/train_XGB.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# Ajuste os imports conforme a sua estrutura
from ML.features import build_pairwise_dataset, EloConfig, cfg as data_cfg
from ML.splits import make_split_plan, split_df_by_fold

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# ----------------------------
# Configuração Blindada
# ----------------------------

@dataclass(frozen=True)
class TrainConfig:
    target_col: str = "target"
    random_state: int = 42
    
    # 🚫 LISTA NEGRA ATUALIZADA
    drop_cols: Tuple[str, ...] = (
        # Metadados
        "date", "year", "tourney_date", "tourney_id", "match_num",
        "winner_id", "loser_id", "p1_id", "p2_id",
        
        # Texto
        "tourney_name", "surface", "round", "tourney_level", 
        "p1_name", "p2_name", "p1_entry", "p2_entry", 
        "p1_hand", "p2_hand", "p1_ioc", "p2_ioc", 
        "score", "minutes",
        
        # 🚨 LEAKAGE ESTRUTURAL DO ELO (O motivo do AUC 1.0)
        "elo_diff_wl",    # Vazamento matemático
        "p1_elo_prob",    # Vazamento por Nulos (só existe pro winner)
        "p2_elo_prob",    # Vazamento por Nulos (só existe pro winner)
        "p1_elo_pre",     # Já usamos elo_diff e elo_prob_p1, o valor bruto pode confundir se mal tratado
        "p2_elo_pre",
        "elo_diff_wl", "p1_elo_prob", "p2_elo_prob", "p1_elo_pre", "p2_elo_pre",
        
        # NOVO: Leakage Surface (Raw values e probabilidades do WL)
        "winner_elo_surface", "loser_elo_surface", # Colunas originais do WL
        "p1_elo_surface", "p2_elo_surface",        # Colunas brutas pairwise (usamos só o diff)
        
        # ... (stats pós jogo mantenha iguais) ...
        
        # Stats pós-jogo
        "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
        "w_1stin", "l_1stin", "w_1stwon", "l_1stwon", 
        "w_2ndwon", "l_2ndwon", "w_svgms", "l_svgms", 
        "w_bpsaved", "l_bpsaved", "w_bpfaced", "l_bpfaced",
        "winner_rank_points", "loser_rank_points",
        "p1_rank_points", "p2_rank_points"
    )

def make_xy(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """Prepara X e y com limpeza agressiva."""
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target col: {cfg.target_col}")

    # Separa Target
    y = df[cfg.target_col].astype(int).to_numpy()
    X = df.drop(columns=[cfg.target_col], errors="ignore")

    # 1. Drop pela Lista Negra (Config)
    X = X.drop(columns=list(cfg.drop_cols), errors="ignore")
    
    # 2. Drop extra para qualquer coluna de stats que tenha escapado
    # (Ex: w_ace que virou w_Ace por causa de maiúsculas)
    leakage_suffixes = ("_ace", "_df", "_svpt", "_1stin", "_1stwon", "_2ndwon", "_svgms", "_bpsaved", "_bpfaced")
    cols_to_drop_extra = [c for c in X.columns if c.lower().endswith(leakage_suffixes) or c.lower().startswith(("w_", "l_"))]
    if cols_to_drop_extra:
        X = X.drop(columns=cols_to_drop_extra)

    # 3. Garante que só sobrou número
    X_num = X.select_dtypes(include=[np.number, bool])
    
    # Aviso se dropou algo útil sem querer
    dropped_types = [c for c in X.columns if c not in X_num.columns]
    if dropped_types:
        print(f"   [make_xy] 🧹 Removendo texto/obj restante: {dropped_types}")

    return X_num, y


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return {
        "logloss": float(log_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    *,
    params: Dict[str, Any],
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200
) -> Tuple[Any, Dict[str, float]]:
    
    # Usa nomes das colunas para facilitar debug
    feature_names = list(X_train.columns)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)

    watchlist = [(dtrain, "train"), (deval, "eval")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False, # Silencioso para não poluir
    )

    best_iter = model.best_iteration
    y_prob = model.predict(deval, iteration_range=(0, best_iter + 1))
    
    metrics = eval_metrics(y_eval, y_prob)
    metrics["best_iteration"] = int(best_iter)
    
    return model, metrics


def print_importances(model, title: str = "Feature Importance"):
    """Imprime as features reais usadas pelo modelo."""
    importance = model.get_score(importance_type='gain')
    if not importance:
        print(f"[{title}] Sem features usadas.")
        return
        
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n🔍 {title} (Top 10 Gain):")
    for feat, val in sorted_importance:
        print(f"   {feat:25s}: {val:.2f}")
    print("-" * 50)


def main():
    print("🚀 Iniciando Pipeline de Treino ATP (Blindado contra Leakage)...")
    
    # 1. Dataset
    # Garante que features.py está corrigido para carregar desde 2010
    elo_cfg = EloConfig(base=1500.0, k=32.0, k_new=64.0, provisional_games=10, add_prob=True)
    df = build_pairwise_dataset(2010, 2026, data_cfg=data_cfg, elo_cfg=elo_cfg)
    print(f"📊 Dataset Carregado: {df.shape}")

    # 2. Configs
    plan = make_split_plan()
    train_cfg = TrainConfig()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "nthread": -1
    }

    # 3. Tuning Loop
    print("\n--- 🔄 TUNING FOLDS ---")
    fold_metrics = []
    
    for fold in plan.tuning_folds:
        print(f"\n>> Fold: {fold.name}")
        tr, ev = split_df_by_fold(df, fold)

        Xtr, ytr = make_xy(tr, train_cfg)
        Xev, yev = make_xy(ev, train_cfg)
        
        # DEBUG CRÍTICO: Imprime as colunas na primeira vez para conferir
        if fold.name == plan.tuning_folds[0].name:
            print(f"👀 Features REAIS entrando no modelo ({len(Xtr.columns)}):")
            print(list(Xtr.columns))
            print("-" * 30)

        model, m = fit_xgb(Xtr, ytr, Xev, yev, params=params)
        fold_metrics.append(m)
        
        print(f"   ✅ LogLoss: {m['logloss']:.5f} | AUC: {m['auc']:.5f}")
        
        # Mostra features do último fold para garantir
        if fold.name == plan.tuning_folds[-1].name:
            print_importances(model, title=f"Importances ({fold.name})")

    avg_loss = np.mean([x['logloss'] for x in fold_metrics])
    print(f"\n📉 Média LogLoss Tuning: {avg_loss:.5f}")

    # 4. Final Val
    print("\n--- 🏆 FINAL VALIDATION (2024) ---")
    tr, ev = split_df_by_fold(df, plan.final_val)
    Xtr, ytr = make_xy(tr, train_cfg)
    Xev, yev = make_xy(ev, train_cfg)
    
    model_val, m_val = fit_xgb(Xtr, ytr, Xev, yev, params=params)
    print(f"   ✅ 2024 LogLoss: {m_val['logloss']:.5f} | AUC: {m_val['auc']:.5f}")
    print_importances(model_val)

    # ... (final do main em ML/train_XGB.py)
    
    # [IMPORTANTE] Salvar o modelo para a visualização carregar depois
    print("\n💾 Salvando modelo final...")
    model_val.save_model("atp_model_v1.json")
    print("✅ Modelo salvo em 'atp_model_v1.json'")
    
if __name__ == "__main__":
    main()