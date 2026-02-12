# ML/visualize.py
"""
Módulo de Visualização e Explainability.
Gera gráficos técnicos e explicativos para interpretação do modelo.
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, calibration_curve

# Configuração global de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

def load_model(path: str) -> xgb.Booster:
    """Carrega um modelo XGBoost salvo em JSON."""
    try:
        model = xgb.Booster()
        model.load_model(path)
        return model
    except Exception as e:
        sys.exit(f"Erro ao carregar modelo '{path}': {e}")

# ==========================================
# Gráficos Técnicos
# ==========================================

def plot_feature_importance(model: xgb.Booster, top_n: int = 15):
    """Plota as features com maior ganho de informação (Gain)."""
    importance = model.get_score(importance_type='gain')
    if not importance:
        print("[AVISO] Nenhuma importância de feature encontrada no modelo.")
        return

    df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
    df_imp = df_imp.sort_values(by='Gain', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Gain', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top {top_n} Features (Information Gain)')
    plt.xlabel('Gain')
    plt.tight_layout()
    plt.show()

def plot_calibration_curve(y_true, y_prob):
    """Plota diagrama de confiabilidade (Calibração)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label='Modelo')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.xlabel('Probabilidade Predita')
    plt.ylabel('Fração de Positivos Reais')
    plt.title('Curva de Calibração')
    plt.legend()
    plt.show()

# ==========================================
# Explainability (Local)
# ==========================================

def plot_match_explanation(model: xgb.Booster, X_row: pd.DataFrame):
    """
    Usa SHAP values nativos (pred_contribs) para explicar uma única predição.
    """
    feature_names = list(X_row.columns)
    dtest = xgb.DMatrix(X_row, feature_names=feature_names)
    
    # SHAP values: o último valor é o bias (base value)
    shap_values = model.predict(dtest, pred_contribs=True)[0]
    contribs = shap_values[:-1]
    
    df_contrib = pd.DataFrame({
        'Feature': feature_names,
        'Impact': contribs,
        'Value': X_row.values.flatten()
    })
    
    # Filtra Top 10 impactos absolutos
    df_contrib['AbsImpact'] = df_contrib['Impact'].abs()
    df_contrib = df_contrib.sort_values('AbsImpact', ascending=False).head(10)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_contrib['Impact']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_contrib['Feature'], df_contrib['Impact'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("Análise de Decisão da Partida (SHAP Values)")
    plt.xlabel("Impacto na Log-Odds (Verde=A favor P1, Vermelho=Contra P1)")
    
    # Anotações dos valores reais
    for bar, val in zip(bars, df_contrib['Value']):
        width = bar.get_width()
        x_pos = width + (0.05 if width > 0 else -0.05)
        ha = 'left' if width > 0 else 'right'
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.2f}", 
                 va='center', ha=ha, fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()