# ATP Tennis Prediction - Machine Learning Pipeline

Este projeto implementa um pipeline de Machine Learning robusto para prever resultados de partidas de ténis da ATP. O sistema utiliza dados históricos de 2010 a 2026, engenharia de features avançada e um modelo XGBoost com validação Walk-Forward.

## Estrutura do Projeto

O código foi refatorado para seguir padrões de engenharia de software.

* **ML/dataio.py**: Ingestão de dados, limpeza e padronização.
* **ML/features.py**: Engenharia de features, cálculo de Elo Rating e transformação Pairwise.
* **ML/splits.py**: Estratégias de validação cruzada temporal (Walk-Forward).
* **ML/train_XGB.py**: Treinamento do modelo, tuning e avaliação.
* **ML/visualize.py**: Geração de gráficos e explainability.
* **ML/eda_check.py**: Utilitário para verificação de dados.

## Metodologia

### Engenharia de Features
O modelo utiliza apenas informações disponíveis **antes** do início da partida:
* **Elo Rating**: Global e por Superfície (Hard, Clay, Grass).
* **Diferenciais**: Rank, Idade e Altura.
* **Metadados**: Superfície, Mão dominante, etc.

### Validação (Walk-Forward)
* **Tuning**: Janelas anuais de 2019 a 2023.
* **Validação Final**: Treino (2010-2023) -> Teste (2024).

## Resultados de Performance

### Histórico de Validação

| Fold (Ano de Teste) | LogLoss | AUC |
| :--- | :--- | :--- |
| **WF 2019** | 0.61353 | 0.71731 |
| **WF 2020** | 0.59786 | 0.73574 |
| **WF 2021** | 0.60718 | 0.72631 |
| **WF 2022** | 0.59547 | 0.74246 |
| **WF 2023** | 0.61500 | 0.71826 |
| **Média Tuning** | **0.60581** | **~0.728** |

### Resultado Final (2024)
* **LogLoss**: 0.60694
* **AUC**: 0.72537
* **BRIER**: 0.20837
* **ECE10**: 0.0000

### Importância das Features (Top 5)
1. elo_prob_p1
2. rank_diff
3. elo_diff
4. elo_diff_surface
5. elo_prob_surface_p1

## Como Executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Pipelines

1. **Verificar integridade dos dados:**
```bash
python -m ML.eda_check
```

2. **Treinar o modelo:**
```bash
python -m ML.train_XGB
```

3. **Visualizar resultados:**
```bash
python -m ML.visualize
```

## Autores
* **Rodrigo Dias de Sousa**
* **Murillo Moraes Giroldo**