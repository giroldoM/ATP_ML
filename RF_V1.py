import pandas as pd
import plotly.express as px

base = pd.read_csv('/Users/murillogiroldo/ATP_ML/data/atp_matches_2013.csv')

show = base.head()
#mostrar quantos valores nulos tem em cada coluna
nulos = base.isnull().sum()
# grafico = px.bar(x=nulos.index, y=nulos.values, labels={'x':'Colunas', 'y':'Quantidade de valores nulos'}, title='Valores Nulos por Coluna')
# grafico.show()

distrib = px.scatter(base, x='winner_rank', y='loser_rank', title='Distribuição dos Ranks dos Jogadores')
distrib.show()

# nomes dos 20 jogadores com rank mais alto 
top_players = base[['winner_name', 'winner_rank']].sort_values(by='winner_rank').drop_duplicates().head(20)
grafico_top = px.bar(top_players, x='winner_name', y='winner_rank',
                        labels={'winner_name':'Nome do Jogador', 'winner_rank':'Rank do Jogador'},
                        title='Top 20 Jogadores com Melhor Rank')
grafico_top.show()

print(show)
print(nulos)