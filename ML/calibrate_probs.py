# calibrate_probs.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

import xgboost as xgb

from ML.features import build_pairwise_dataset, EloConfig, DataIOConfig
from ML.splits import make_split_plan, split_df_by_fold
from ML.train_XGB import TrainConfig, prepare_xy


def ece10(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) com 10 bins."""
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        acc = float(y_true[m].mean())
        conf = float(y_prob[m].mean())
        ece += abs(acc - conf) * float(m.mean())
    return float(ece)


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "logloss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece10": float(ece10(y, p, 10)),
    }


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="atp_model_v1.json")
    ap.add_argument("--out_path", type=str, default="Betting/calibrator_v1.joblib")
    ap.add_argument("--mode", choices=["auto", "isotonic", "platt"], default="auto")
    args = ap.parse_args()

    if joblib is None:
        raise RuntimeError("joblib não está disponível. Rode: pip install joblib")

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    # 1) Reconstroi o dataset com todas as features recentes (incluindo Stats Avançadas)
    data_cfg = DataIOConfig(data_dir=data_dir, drop_leakage=False, remove_walkovers=True)
    elo_cfg = EloConfig(k=32.0, k_new=64.0, add_prob=True)

    print("📦 Montando dataset pairwise para Calibração (2010–2026)...")
    df = build_pairwise_dataset(2010, 2026, data_cfg=data_cfg, elo_cfg=elo_cfg)

    plan = make_split_plan()
    train_cfg = TrainConfig()

    tr, ev = split_df_by_fold(df, plan.final_val)
    if ev.empty:
        raise RuntimeError("Dataset de Validação (2024) vazio. Verifique se os dados estão na pasta.")

    # [SEGURANÇA] Corta o dataset em dois para o calibrador não ver exatamente o mesmo ruído do early stopping
    calib_start = len(ev) // 2
    ev_calib = ev.iloc[calib_start:].copy()

    Xev, yev = prepare_xy(ev_calib, train_cfg)

    # 3) Carrega modelo e força o alinhamento exato de features
    print(f"🧠 A carregar modelo: {args.model_path}")
    booster = xgb.Booster()
    booster.load_model(args.model_path)

    expected = booster.feature_names
    if expected is None:
        raise RuntimeError("O modelo carregado não possui 'feature_names'. Treine novamente com DMatrix.")

    # Diagnóstico para garantir que as novas features entraram
    advanced_feats = ["fatigue_diff", "serve_pct_diff", "h2h_saldo"]
    detected = [f for f in advanced_feats if f in expected]
    if detected:
        print(f"   [OK] Features avançadas detectadas no modelo: {detected}")

    extra = sorted(set(Xev.columns) - set(expected))
    missing = sorted(set(expected) - set(Xev.columns))

    if extra:
        print(f"   [AVISO] Features ignoradas pelo calibrador (não estão no modelo): {extra}")
    if missing:
        print(f"   [AVISO] Features ausentes no dataset (preenchidas com NaN): {missing}")
        for c in missing:
            Xev[c] = np.nan

    # Garante a ordem e exatidão das colunas
    Xev = Xev.reindex(columns=list(expected))

    dmat = xgb.DMatrix(Xev, feature_names=list(expected))
    p_raw = booster.predict(dmat)
    p_raw = np.clip(p_raw, 1e-9, 1 - 1e-9)

    print("\n== PERFORMANCE RAW (Sem Calibração) ==")
    print(metrics(yev, p_raw))

    cand: Dict[str, Any] = {}

    # 4) Isotonic Regression
    if args.mode in ("auto", "isotonic"):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, yev.astype(int))
        p_iso = np.clip(iso.predict(p_raw), 1e-9, 1 - 1e-9)
        cand["isotonic"] = {"method": "isotonic", "model": iso, "metrics": metrics(yev, p_iso)}

    # 5) Platt Scaling (Logistic Regression)
    if args.mode in ("auto", "platt"):
        Z = logit(p_raw).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
        lr.fit(Z, yev.astype(int))
        p_platt = np.clip(lr.predict_proba(Z)[:, 1], 1e-9, 1 - 1e-9)
        cand["platt"] = {"method": "platt", "model": lr, "metrics": metrics(yev, p_platt)}

    # Seleciona o método que minimizou o LogLoss na base de calibração
    best_name = min(cand.keys(), key=lambda k: cand[k]["metrics"]["logloss"])
    best = cand[best_name]

    print(f"\n== MELHOR MÉTODO: {best_name.upper()} ==")
    for k, v in best["metrics"].items():
        print(f"   {k}: {v:.4f}")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"method": best["method"], "model": best["model"]}, out)
    print(f"\n✅ Calibrador treinado e guardado em: {out}")


if __name__ == "__main__":
    main()