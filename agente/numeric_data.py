"""
Carregamento dos dados numéricos (planilha Av_Desempenho_Renove.xlsx) com pandas.

A planilha tem uma estrutura pouco padronizada (blocos mensais soltos, nomes em
branco em algumas linhas, colunas repetidas por loja). Em vez de codificar regras
de limpeza rígidas — frágeis a qualquer mudança de layout —, carregamos cada aba
como está e damos ao Gemini um resumo do esquema (colunas + amostra de linhas) pra
ele mesmo gerar a expressão pandas correta em tempo de pergunta (ver router.py).

Uso:
    from numeric_data import carregar_dataframes, resumo_esquema
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PLANILHA = BASE_DIR / "dados" / "Av_Desempenho_Renove.xlsx"

# Nomes de abas normalizados para chaves de dicionário válidas em Python
# (sem espaços/acentos), usados como nome de variável disponível pro Gemini.
_MAPA_NOMES = {
    "BASE": "base",
    "META": "meta",
    "REFERENCIA PML": "referencia_pml",
    "CONTROLE": "controle",
}


def carregar_dataframes() -> dict:
    """
    Retorna um dicionário {nome_variavel: DataFrame} com todas as abas da planilha.
    Abas de operadoras individuais (Rafael, Camila, etc.) e a de consolidado por
    operador entram com o nome da aba em minúsculo e sem espaços.
    """
    xls = pd.ExcelFile(PLANILHA)
    dataframes = {}

    for aba in xls.sheet_names:
        df = xls.parse(aba, header=None)
        # Remove linhas e colunas 100% vazias, mas preserva a estrutura de blocos
        df = df.dropna(how="all").dropna(axis=1, how="all")

        chave = _MAPA_NOMES.get(aba)
        if chave is None:
            chave = aba.strip().lower().replace(" ", "_").replace("—", "").replace("-", "_")
            chave = chave.strip("_")

        dataframes[chave] = df

    return dataframes


def resumo_esquema(dataframes: dict, linhas_amostra: int = 6) -> str:
    """
    Gera um resumo em texto do esquema de cada DataFrame (nome, dimensões e
    amostra de linhas), pra dar contexto ao Gemini na hora de gerar a consulta.
    """
    partes = []
    for nome, df in dataframes.items():
        partes.append(f"### DataFrame `{nome}` ({df.shape[0]} linhas x {df.shape[1]} colunas)")
        amostra = df.head(linhas_amostra).to_string(index=True, max_colwidth=25)
        partes.append(amostra)
        partes.append("")
    return "\n".join(partes)


if __name__ == "__main__":
    dfs = carregar_dataframes()
    print(f"Abas carregadas: {list(dfs.keys())}\n")
    print(resumo_esquema(dfs, linhas_amostra=3))
