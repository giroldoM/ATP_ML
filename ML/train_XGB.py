# ML/train_xgb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

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
    drop_cols: Tuple[str, ...] = ("date", "year")  # year você cria no splits.py
    # se tiver alguma coluna “texto” que escapou (ex.: surface), você ou encoda antes ou dropa aqui
    # drop_cols pode ser aumentado quando você ver as colunas finais
    random_state: int = 42


def make_xy(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target col: {cfg.target_col}")

    y = df[cfg.target_col].astype(int).to_numpy()
    X = df.drop(columns=[cfg.target_col], errors="ignore")

    # Drop colunas não-feature
    X = X.drop(columns=list(cfg.drop_cols), errors="ignore")

    # Remove colunas object automaticamente (pra não quebrar o xgboost)
    # (Depois você pode trocar por encoding/categoricals)
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        X = X.drop(columns=obj_cols)

    return X, y


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    # y_prob deve ser prob de classe 1
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
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
        verbose_eval=200,
    )

    y_prob = model.predict(deval, iteration_range=(0, model.best_iteration + 1))
    metrics = eval_metrics(y_eval, y_prob)
    metrics["best_iteration"] = int(model.best_iteration)
    return model, metrics


def main():
    # 1) Build dataset com Elo
    elo_cfg = EloConfig(base=1500.0, k=32.0, k_new=64.0, provisional_games=10, add_prob=True)

    # Ex: gerar até 2026 de uma vez, pq Elo precisa do histórico
    df = build_pairwise_dataset(
        2010, 2026,
        data_cfg=data_cfg,
        elo_cfg=elo_cfg
    )

    # 2) Split plan
    plan = make_split_plan()

    # 3) Params baseline do XGB (simples e bom pra começar)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
        "tree_method": "hist",
    }

    train_cfg = TrainConfig()

    # 4) Tuning folds (2019–2023)
    fold_results: List[Dict[str, Any]] = []
    for fold in plan.tuning_folds:
        tr, ev = split_df_by_fold(df, fold)

        Xtr, ytr = make_xy(tr, train_cfg)
        Xev, yev = make_xy(ev, train_cfg)

        model, m = fit_xgb(Xtr, ytr, Xev, yev, params=params)
        fold_results.append({"fold": fold.name, **m})
        print(f"[{fold.name}] {m}")

    # 5) Final val (treina 2010–2023, valida 2024)
    tr, ev = split_df_by_fold(df, plan.final_val)
    Xtr, ytr = make_xy(tr, train_cfg)
    Xev, yev = make_xy(ev, train_cfg)
    model_val, m_val = fit_xgb(Xtr, ytr, Xev, yev, params=params)
    print("\n[FINAL_VAL]", m_val)

    # 6) Test (treina 2010–2024, testa 2025–2026)
    tr, ev = split_df_by_fold(df, plan.test)
    Xtr, ytr = make_xy(tr, train_cfg)
    Xev, yev = make_xy(ev, train_cfg)
    model_test, m_test = fit_xgb(Xtr, ytr, Xev, yev, params=params)
    print("\n[TEST]", m_test)


if __name__ == "__main__":
    main()
