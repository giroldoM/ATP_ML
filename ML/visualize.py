# ML/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

# Importa as configs do seu pipeline
from ML.features import build_pairwise_dataset, EloConfig, cfg as data_cfg
from ML.splits import make_split_plan, split_df_by_fold
# Importa a função de preparação de dados do treino
from ML.train_XGB import make_xy, TrainConfig

# ==========================================
# GRÁFICOS TÉCNICOS (Para Devs/Data Scientists)
# ==========================================

def plot_feature_importance(model, feature_names=None, top_n=15):
    """Plota as features mais importantes (Gain)."""
    importance = model.get_score(importance_type='gain')
    if not importance:
        print("⚠️ Nenhuma feature importance encontrada.")
        return

    df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
    df_imp = df_imp.sort_values(by='Gain', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x='Gain', y='Feature', palette='viridis')
    plt.title(f'Top {top_n} Features (Importância Técnica)')
    plt.xlabel('Ganho de Informação')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob):
    """Plota a Curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Performance do Modelo (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_calibration_curve(y_true, y_prob):
    """Plota a Curva de Calibração (Reliability Diagram)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Seu Modelo')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfeição')
    plt.xlabel('Confiança do Modelo')
    plt.ylabel('Realidade (Vitórias Reais)')
    plt.title('Calibração: O modelo sabe o que diz?')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ==========================================
# GRÁFICOS EXPLICATIVOS (Para Leigos)
# ==========================================

def plot_match_explanation(model, X_row, feature_names):
    """
    Mostra os 'Prós' e 'Contras' de UMA partida específica.
    Usa 'pred_contribs' (SHAP nativo do XGBoost) para explicar a decisão.
    """
    # Cria DMatrix para uma única linha
    dtest = xgb.DMatrix(X_row, feature_names=feature_names)
    
    # pred_contribs=True retorna o valor base + contribuição de cada feature
    # O último valor do array é o Bias (valor base), ignoramos ele para o gráfico
    contribs = model.predict(dtest, pred_contribs=True)[0][:-1]
    
    # Cria DataFrame para plotar
    df_contrib = pd.DataFrame({
        'Feature': feature_names,
        'Impacto': contribs,
        'Valor': X_row.values.flatten()
    })
    
    # Filtra as features que tiveram impacto relevante (top 10 absoluto)
    df_contrib['AbsImpact'] = df_contrib['Impacto'].abs()
    df_contrib = df_contrib.sort_values('AbsImpact', ascending=False).head(10)
    
    # Cores: Verde (Aumentou chance de vitória) / Vermelho (Diminuiu)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_contrib['Impacto']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_contrib['Feature'], df_contrib['Impacto'], color=colors)
    
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f"Por que a IA apostou assim? (Anatomia da Decisão)\nVerde = Ajudou o Jogador 1 | Vermelho = Atrapalhou")
    plt.xlabel("Impacto na Probabilidade (Log-Odds)")
    
    # Adiciona o valor real da feature nas barras
    for bar, val, feat in zip(bars, df_contrib['Valor'], df_contrib['Feature']):
        x_pos = bar.get_width()
        align = 'left' if x_pos > 0 else 'right'
        offset = 0.05 if x_pos > 0 else -0.05
        plt.text(x_pos + offset, bar.get_y() + bar.get_height()/2, 
                 f"{val:.2f}", va='center', ha=align, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_single_tree(model):
    """
    Tenta plotar a estrutura de UMA árvore de decisão.
    Requer graphviz instalado no sistema. Se falhar, avisa.
    """
    try:
        plt.figure(figsize=(20, 12))
        # Plota a árvore #0 (a primeira de milhares)
        xgb.plot_tree(model, num_trees=0, rankdir='LR') 
        plt.title("O 'Cérebro' da IA: Exemplo de uma das 1000 árvores de decisão")
        plt.show()
    except Exception as e:
        print(f"\n⚠️ Não foi possível gerar o gráfico da árvore (Graphviz ausente?).\nErro: {e}")
        print("Dica: Para ver a árvore, instale 'graphviz' no sistema (brew install graphviz).")


def main():
    print("🚀 Iniciando Visualização...")

    # 1. Configs e Dados
    elo_cfg = EloConfig(base=1500.0, k=32.0, k_new=64.0, provisional_games=10, add_prob=True)
    df = build_pairwise_dataset(2010, 2026, data_cfg=data_cfg, elo_cfg=elo_cfg)
    
    plan = make_split_plan()
    train_cfg = TrainConfig()

    # 2. Carregar Modelo
    model_path = "atp_model_v1.json"
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"📥 Modelo carregado: {model_path}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return

    # 3. Dados de Validação
    _, ev = split_df_by_fold(df, plan.final_val)
    X_ev, y_ev = make_xy(ev, train_cfg)
    
    feature_names = list(X_ev.columns)
    dtest = xgb.DMatrix(X_ev, feature_names=feature_names)

    # 4. Previsões Gerais
    y_prob = model.predict(dtest)

    # ==========================
    # VISUALIZAÇÃO
    # ==========================
    
    # 1. Gráficos Técnicos (Para você e seu amigo)
    print("\n📊 Gerando gráficos técnicos...")
    plot_feature_importance(model, top_n=10)
    plot_calibration_curve(y_ev, y_prob)
    
    # 2. Gráficos para Leigos (Storytelling)
    print("\n📖 Gerando explicação para leigos...")
    
    # A. Visualizar a Árvore (O Processo)
    # Isso mostra que não é mágica, é um fluxograma gigante
    plot_single_tree(model)
    
    # B. Explicar UMA partida específica (O Raciocínio)
    # Escolhemos uma partida onde o modelo teve alta confiança (>80%) para ficar claro
    high_conf_idx = np.where(y_prob > 0.85)[0]
    if len(high_conf_idx) > 0:
        idx = high_conf_idx[0] # Pega a primeira partida "óbvia" que achou
        
        # Recupera os dados dessa linha
        X_single = X_ev.iloc[[idx]]
        
        print(f"\n🔎 Explicando a partida índice {idx} (Probabilidade P1: {y_prob[idx]:.2%})")
        plot_match_explanation(model, X_single, feature_names)
    else:
        # Se não tiver nenhuma muito óbvia, pega a primeira mesmo
        print("\n🔎 Explicando a primeira partida do dataset...")
        plot_match_explanation(model, X_ev.iloc[[0]], feature_names)

    print("✅ Concluído!")

if __name__ == "__main__":
    main()