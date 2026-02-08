import pandas as pd
import plotly.express as px

base = pd.read_csv('/Users/murillogiroldo/ATP_ML/data/atp_matches_2010.csv')

show = base.head()
#mostrar quantos valores nulos tem em cada coluna
nulos = base.isnull().sum()
# # grafico = px.bar(x=nulos.index, y=nulos.values, labels={'x':'Colunas', 'y':'Quantidade de valores nulos'}, title='Valores Nulos por Coluna')
# # grafico.show()

# distrib = px.scatter(base, x='winner_rank', y='loser_rank', title='Distribuição dos Ranks dos Jogadores')
# distrib.show()

# # nomes dos 20 jogadores com rank mais alto 
# top_players = base[['winner_name', 'winner_rank']].sort_values(by='winner_rank').drop_duplicates().head(20)
# grafico_top = px.bar(top_players, x='winner_name', y='winner_rank',
#                         labels={'winner_name':'Nome do Jogador', 'winner_rank':'Rank do Jogador'},
#                         title='Top 20 Jogadores com Melhor Rank')
# grafico_top.show()

# print(show)
def dataframe(ano) -> pd.DataFrame:
    df = pd.read_csv(f'/Users/murillogiroldo/ATP_ML/data/atp_matches_{ano}.csv')
    return df
#mostrar quantos valores nulos tem em colunas vitais[surface, round, winner_id, loser_id, indoor, tourney level]
def colunas_vitais(df: pd.DataFrame) -> pd.Series:
    vitais = ['surface', 'round', 'winner_id', 'loser_id', 'indoor', 'tourney_level']

    # pega só as colunas que existem nesse arquivo
    presentes = [c for c in vitais if c in df.columns]
    faltando  = [c for c in vitais if c not in df.columns]

    nulos = df[presentes].isnull().sum()

    # opcional: registrar as que nem existem (aparece com NaN ou -1)
    for c in faltando:
        nulos[c] = pd.NA  # ou 0, ou -1, depende do que você quer mostrar

    # reordena pra sempre imprimir na mesma ordem
    return nulos[vitais]


def pronto(ano_inicial, ano_final):
    for ano in range(ano_inicial, ano_final + 1):
        df = dataframe(ano)
        nulos_vitais = colunas_vitais(df)
        print(f'Valores nulos nas colunas vitais para o ano {ano}:')
        print(nulos_vitais)
        print('---')

pronto(2010, 2026)



# import pandas as pd

# PATH = "/Users/murillogiroldo/ATP_ML/data/atp_matches_2023.csv"

# SURFACE_FIX = {
#     "Davis Cup WG1 PO: BRA vs CHN": "Clay",
#     "Davis Cup WG1 PO: ISR vs LAT": "Hard",
#     "Davis Cup WG1 PO: PAK vs LTU": "Hard",
#     "Davis Cup WG1 PO: PER vs IRL": "Clay",
#     "Davis Cup WG1 PO: ROU vs THA": "Hard",
#     "Davis Cup WG1 PO: TUR vs SLO": "Hard",
#     "Davis Cup WG1 PO: UKR vs LBN": "Hard",
#     "Davis Cup WG2 PO: BAR vs POC": "Hard",
#     "Davis Cup WG2 PO: BOL vs GEO": "Hard",
#     "Davis Cup WG2 PO: EGY vs PAR": "Clay",
#     "Davis Cup WG2 PO: ESA vs JOR": "Hard",
#     "Davis Cup WG2 PO: EST vs JAM": "Hard",
#     "Davis Cup WG2 PO: INA vs VIE": "Hard",
#     "Davis Cup WG2 PO: RSA vs LUX": "Hard",
#     "Davis Cup WG2 PO: TUN vs CYP": "Hard",
# }

# df = pd.read_csv(PATH)

# # normaliza o tourney_name pra bater certinho (remove espaços, NBSP etc.)
# tname = (
#     df["tourney_name"]
#     .astype(str)
#     .str.replace("\xa0", " ", regex=False)  # NBSP
#     .str.strip()
# )

# # máscara: linhas cujo tourney_name está no dicionário
# mask = tname.isin(SURFACE_FIX.keys())

# # aplica a superfície correspondente (sobrescreve mesmo)
# df.loc[mask, "surface"] = tname[mask].map(SURFACE_FIX)

# print("Linhas afetadas (tourney_name bateu):", int(mask.sum()))
# print("NaN em surface depois:", int(df["surface"].isna().sum()))
# print("Surface vazia depois:", int(df["surface"].astype(str).str.strip().eq("").sum()))

# df.to_csv(PATH, index=False)
# print("Arquivo sobrescrito:", PATH)
