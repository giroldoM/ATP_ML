# ML/train_XGB.py
"""
Script de Treinamento do Modelo XGBoost.
Executa o loop de validação (Tuning), treina o modelo final e salva os artefatos.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

try:
    import xgboost as xgb
except ImportError:
    sys.exit("Erro Critico: Biblioteca 'xgboost' não encontrada. Instale com pip install xgboost.")

# Imports locais (assumindo execução como módulo: python -m ML.train_XGB)
from ML.features import build_pairwise_dataset, EloConfig, DataIOConfig
from ML.splits import make_split_plan, split_df_by_fold
import optuna
# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@dataclass(frozen=True)
class TrainConfig:
    target_col: str = "target"
    random_state: int = 42
    
    # Blacklist: Colunas que JAMAIS devem entrar no treino
    # Inclui metadados, strings e features com vazamento de dados (leakage)
    cols_to_drop: Tuple[str, ...] = (
        "date", "year", "tourney_date", "tourney_id", "match_num",
        "winner_id", "loser_id", "p1_id", "p2_id",
        "tourney_name", "surface", "round", "tourney_level", 
        "p1_name", "p2_name", "p1_entry", "p2_entry", 
        "p1_hand", "p2_hand", "p1_ioc", "p2_ioc", 
        "score", "minutes",
        # Leakage de Elo (valores brutos e colunas originais do WL)
        "winner_elo_pre", "loser_elo_pre",
        "winner_elo_surface", "loser_elo_surface",
        "p1_elo_pre", "p2_elo_pre",
        "p1_elo_surface", "p2_elo_surface",
    )

# -----------------------------------------------------------------------------
# Helpers de Modelagem
# -----------------------------------------------------------------------------

def prepare_xy(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Separa features (X) e target (y), aplicando limpeza rigorosa de colunas proibidas.
    """
    if cfg.target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{cfg.target_col}' não encontrada no dataset.")

    y = df[cfg.target_col].astype(int).to_numpy()
    X = df.drop(columns=[cfg.target_col])

    # 1. Remove colunas da Blacklist
    X = X.drop(columns=list(cfg.cols_to_drop), errors="ignore")
    
    # 2. Varredura de Segurança: Remove colunas residuais de estatísticas (w_ace, etc)
    # Caso alguma tenha escapado do filtro anterior
    forbidden_substrings = ["_ace", "_df", "_svpt", "_1st", "_2nd", "_svgms", "_bp"]
    leakage_cols = [c for c in X.columns if any(sub in c.lower() for sub in forbidden_substrings)]
    if leakage_cols:
        X = X.drop(columns=leakage_cols)

    # 3. Garante apenas dados numéricos/booleanos
    X_clean = X.select_dtypes(include=[np.number, bool])
    
    if len(X_clean.columns) < len(X.columns):
        removed = set(X.columns) - set(X_clean.columns)
        print(f"   [AVISO] Colunas não-numéricas removidas automaticamente: {removed}")

    return X_clean, y

def train_model(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_eval: pd.DataFrame, y_eval: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[xgb.Booster, Dict[str, float]]:
    """Treina o modelo XGBoost com early stopping."""
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=list(X_train.columns))

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # Previsão na melhor iteração
    y_prob = model.predict(deval, iteration_range=(0, model.best_iteration + 1))
    
    metrics = {
        "logloss": log_loss(y_eval, y_prob),
        "auc": roc_auc_score(y_eval, y_prob),
        "brier": brier_score_loss(y_eval, y_prob),
        "best_iter": model.best_iteration
    }
    
    return model, metrics

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------




def main():
    print("[INFO] ATP Tennis Machine Learning Pipeline com Optuna Tuning")
    print("==============================================================")

    print("\n[1] Carregando e processando dados...")
    data_cfg = DataIOConfig(data_dir=DATA_DIR, drop_leakage=False, remove_walkovers=True)
    elo_cfg = EloConfig(k=32.0, k_new=64.0, add_prob=True)
    
    df = build_pairwise_dataset(2010, 2024, data_cfg=data_cfg, elo_cfg=elo_cfg)
    plan = make_split_plan()
    train_cfg = TrainConfig()

    # Função objetivo para o Optuna
    def objective(trial):
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "seed": 42,
            "nthread": -1
        }
        
        fold_losses = []
        # Usa os mesmos folds de tuning que você já tinha configurado
        for fold in plan.tuning_folds:
            train_df, eval_df = split_df_by_fold(df, fold)
            X_tr, y_tr = prepare_xy(train_df, train_cfg)
            X_ev, y_ev = prepare_xy(eval_df, train_cfg)

            _, metrics = train_model(X_tr, y_tr, X_ev, y_ev, xgb_params)
            
            # Aqui estamos minimizando o Brier Score, já que é seu alvo principal de melhoria
            fold_losses.append(metrics['brier']) 

        return np.mean(fold_losses)

    print("\n[2] Iniciando Busca de Hiperparâmetros (Optuna)...")
    # Cria o estudo buscando minimizar o Brier Score
    study = optuna.create_study(direction="minimize")
    # n_trials=30 é um bom começo. Para rodar a noite, coloque 100 ou 200.
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n   [RESUMO OPTUNA] Melhores Parâmetros Encontrados:")
    best_params = study.best_params
    best_params.update({"objective": "binary:logistic", "eval_metric": "logloss", "seed": 42, "nthread": -1})
    for k, v in best_params.items():
        print(f"      {k}: {v}")
    print(f"      Melhor Brier Score Médio na Validação: {study.best_value:.4f}")

    # 4. Treino Final com os parâmetros tunados
    print("\n[3] Treinando Modelo Final (Validado no fold final)...")
    tr_final, ev_final = split_df_by_fold(df, plan.final_val)
    X_tr_f, y_tr_f = prepare_xy(tr_final, train_cfg)
    X_ev_f, y_ev_f = prepare_xy(ev_final, train_cfg)

    model_final, metrics_final = train_model(X_tr_f, y_tr_f, X_ev_f, y_ev_f, best_params)
    
    print(f"   [RESULTADO] Final: LogLoss {metrics_final['logloss']:.4f} | AUC {metrics_final['auc']:.4f} | Brier {metrics_final['brier']:.4f}")
    
    output_path = "atp_model_v1.json"
    model_final.save_model(output_path)
    print(f"\n[SUCESSO] Modelo salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    main()