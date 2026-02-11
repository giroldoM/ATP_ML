# ML/train_XGB.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# Certifique-se que estes imports apontam para os arquivos corrigidos
from ML.features import build_pairwise_dataset, EloConfig, cfg as data_cfg
from ML.splits import make_split_plan, split_df_by_fold

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class TrainConfig:
    target_col: str = "target"
    # Adicione aqui colunas que sobraram e não devem ir pro treino
    drop_cols: Tuple[str, ...] = ("date", "year", "tourney_date") 
    random_state: int = 42


def make_xy(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target col: {cfg.target_col}")

    y = df[cfg.target_col].astype(int).to_numpy()
    X = df.drop(columns=[cfg.target_col], errors="ignore")

    # Drop colunas explícitas (metadados)
    X = X.drop(columns=list(cfg.drop_cols), errors="ignore")

    # ⚠️ Segurança: Remove colunas object automaticamente, MAS AVISA
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        print(f"   [make_xy] ⚠️ Dropando colunas object (não-codificadas): {obj_cols}")
        X = X.drop(columns=obj_cols)

    return X, y


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    # Clip para evitar log(0)
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
    early_stopping_rounds: int = 200,
    verbose_eval: int = 200
) -> Tuple[Any, Dict[str, float]]:
    if xgb is None:
        raise ImportError("xgboost não está instalado. pip install xgboost")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    watchlist = [(dtrain, "train"), (deval, "eval")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )

    # Melhor iteração
    best_iter = model.best_iteration
    # Previsão na validação
    y_prob = model.predict(deval, iteration_range=(0, best_iter + 1))
    
    metrics = eval_metrics(y_eval, y_prob)
    metrics["best_iteration"] = int(best_iter)
    
    return model, metrics


def print_importances(model, title: str = "Feature Importance"):
    """Imprime as top 10 features por Ganho (Gain)"""
    importance = model.get_score(importance_type='gain')
    if not importance:
        print(f"[{title}] Sem features usadas.")
        return
        
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n🔍 {title} (Top 10 Gain):")
    for feat, val in sorted_importance:
        print(f"   {feat:20s}: {val:.2f}")
    print("-" * 40)


def main():
    print("🚀 Iniciando Pipeline de Treino ATP...")
    
    # 1) Build dataset com Elo (carrega histórico 2010+)
    elo_cfg = EloConfig(base=1500.0, k=32.0, k_new=64.0, provisional_games=10, add_prob=True)
    
    # Gera até o futuro (2026) para ter os dados de teste prontos
    df = build_pairwise_dataset(
        2010, 2026,
        data_cfg=data_cfg,
        elo_cfg=elo_cfg
    )
    print(f"📊 Dataset Completo: {df.shape}")

    # 2) Split plan
    plan = make_split_plan()
    train_cfg = TrainConfig()

    # 3) Params XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,               # Um pouco mais alto que 0.03 pra ser mais rápido no teste
        "max_depth": 4,            # Evita overfitting em dataset tabular ruidoso
        "min_child_weight": 10,    # Conservador
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.5,
        "alpha": 0.1,
        "seed": 42,
        "tree_method": "hist",     # Mais rápido
    }

    # 4) Tuning folds (Walk-Forward Validation)
    print("\n--- 🔄 TUNING FOLDS (Walk-Forward) ---")
    fold_metrics = []
    
    for fold in plan.tuning_folds:
        print(f"\n>> Fold: {fold.name}")
        tr, ev = split_df_by_fold(df, fold)

        Xtr, ytr = make_xy(tr, train_cfg)
        Xev, yev = make_xy(ev, train_cfg)
        
        # Sanity check nas dimensões
        print(f"   Train: {Xtr.shape}, Eval: {Xev.shape}")

        model, m = fit_xgb(Xtr, ytr, Xev, yev, params=params, verbose_eval=False)
        fold_metrics.append(m)
        
        print(f"   ✅ LogLoss: {m['logloss']:.5f} | AUC: {m['auc']:.5f} | Best Iter: {m['best_iteration']}")
        
        # Opcional: ver features de um dos folds para checar leakage cedo
        if fold.name == plan.tuning_folds[-1].name:
            print_importances(model, title=f"Importances ({fold.name})")

    avg_logloss = np.mean([x['logloss'] for x in fold_metrics])
    print(f"\n📉 Média Tuning LogLoss: {avg_logloss:.5f}")

    # 5) Final Val (Simulação Real de 2024)
    print("\n--- 🏆 FINAL VALIDATION (2024) ---")
    tr, ev = split_df_by_fold(df, plan.final_val)
    Xtr, ytr = make_xy(tr, train_cfg)
    Xev, yev = make_xy(ev, train_cfg)
    
    model_val, m_val = fit_xgb(Xtr, ytr, Xev, yev, params=params)
    print(f"   ✅ LogLoss: {m_val['logloss']:.5f} | AUC: {m_val['auc']:.5f}")
    print_importances(model_val, title="Final Val Importances")

    # 6) Test (O Futuro - 2025/2026)
    # Nota: Como ainda não temos resultados de 2026, isso aqui serve para gerar o modelo
    # que será usado em produção. A métrica de teste só será real se o CSV tiver resultados.
    print("\n--- 🔮 TEST / PROD MODEL (2025-2026) ---")
    tr, ev = split_df_by_fold(df, plan.test)
    Xtr, ytr = make_xy(tr, train_cfg)
    
    # Se 2025/2026 tiver targets (jogos já ocorridos), avaliamos.
    # Se for futuro puro (sem target), treinamos no dataset todo para salvar o modelo.
    if ev[train_cfg.target_col].notna().any():
        Xev, yev = make_xy(ev, train_cfg)
        model_test, m_test = fit_xgb(Xtr, ytr, Xev, yev, params=params)
        print(f"   ✅ Test LogLoss: {m_test['logloss']:.5f}")
    else:
        print("   ⚠️ Dataset de teste sem targets (futuro). Treinando modelo final no dataset todo disponível...")
        # Aqui você poderia treinar com tudo e salvar o modelo
        pass


if __name__ == "__main__":
    main()