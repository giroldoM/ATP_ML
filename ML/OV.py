import pandas as pd
import plotly.express as px

base = pd.read_csv('/Users/murillogiroldo/ATP_ML/data/atp_matches_2010.csv')

show = base.head()
#mostrar quantos valores nulos tem em cada coluna
nulos = base.isnull().sum()


# print(show)print(nulos)



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

a = 2 **220 
print(a)