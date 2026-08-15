"""
Orquestrador do agente RAG da Renove Lavanderias.

Fluxo de cada pergunta:
  1. Classifica a intenção: pergunta de regra/processo/treinamento (documentos)
     ou pergunta numérica (planilha de desempenho)?
  2a. Se "documentos": busca semântica no índice FAISS (retrieval.py) + geração
      de resposta com Gemini, citando a fonte, restrita ao contexto recuperado.
  2b. Se "numerico": Gemini gera uma expressão pandas sobre os DataFrames da
      planilha (numeric_data.py), executada num ambiente restrito, e o
      resultado é formatado em linguagem natural.
  3. Se nada relevante for encontrado, o agente informa isso explicitamente
     em vez de arriscar uma resposta incorreta (nunca inventa dado numérico
     nem cita política que não está nos documentos).

Usa o SDK atual "google-genai" (não "google-generativeai", descontinuado).

Uso (via Streamlit, ver app.py):
    from rag_agent import responder_pergunta
    resultado = responder_pergunta("Quantos atrasos de até 5 min são tolerados?")
"""

import json
import os

import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

import numeric_data
import retrieval

load_dotenv()

GERACAO_MODEL = "gemini-3.6-flash"

_cliente_cache = None
_dataframes_cache = None
_schema_cache = None


def _cliente() -> genai.Client:
    global _cliente_cache
    if _cliente_cache is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY não encontrada. Configure o .env local ou os Secrets "
                "do Streamlit Cloud (veja .env.example)."
            )
        _cliente_cache = genai.Client(api_key=api_key)
    return _cliente_cache


def _dataframes():
    global _dataframes_cache, _schema_cache
    if _dataframes_cache is None:
        _dataframes_cache = numeric_data.carregar_dataframes()
        _schema_cache = numeric_data.resumo_esquema(_dataframes_cache)
    return _dataframes_cache, _schema_cache


# ---------------------------------------------------------------------------
# 1. Classificação de intenção
# ---------------------------------------------------------------------------

PROMPT_ROTEADOR = """Você classifica perguntas de colaboradores de uma empresa de lavanderia.

Responda APENAS com um JSON no formato: {{"rota": "documentos"}} ou {{"rota": "numerico"}}

- "numerico": a pergunta pede um número, valor, quantidade, comparação ou cálculo
  vindo dos dados de desempenho (ex.: pacotes vendidos, faturamento, premiação em R$,
  % de meta atingida, comissão de uma operadora, dados de uma loja específica).
- "documentos": a pergunta é sobre regras, processos, políticas, como fazer algo,
  ou conhecimento técnico (ex.: regras de assiduidade, como extrair um relatório,
  como cuidar de um tipo de tecido).

Pergunta: {pergunta}
"""


def _classificar_intencao(cliente: genai.Client, pergunta: str) -> str:
    resposta = cliente.models.generate_content(
        model=GERACAO_MODEL,
        contents=PROMPT_ROTEADOR.format(pergunta=pergunta),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    try:
        dados = json.loads(resposta.text)
        rota = dados.get("rota", "documentos")
    except (json.JSONDecodeError, AttributeError, TypeError):
        rota = "documentos"
    return rota if rota in ("documentos", "numerico") else "documentos"


# ---------------------------------------------------------------------------
# 2a. Rota de documentos (RAG clássico)
# ---------------------------------------------------------------------------

PROMPT_RESPOSTA_DOCUMENTOS = """Você é o agente interno de conhecimento da Renove Lavanderias.
Responda a pergunta do colaborador usando SOMENTE as informações do CONTEXTO abaixo.
Não use conhecimento externo. Se o contexto não for suficiente para responder com
segurança, diga claramente que não encontrou essa informação nos documentos disponíveis.

Ao final da resposta, cite a(s) fonte(s) usada(s) entre parênteses (nome do arquivo).

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}
"""


def _responder_via_documentos(cliente: genai.Client, pergunta: str) -> dict:
    trechos = retrieval.buscar_trechos_relevantes(cliente, pergunta, k=4)

    if not trechos:
        return {
            "resposta": (
                "Não encontrei essa informação nos documentos disponíveis. "
                "Pode ser útil falar com a área responsável (RH ou gestão) pra confirmar."
            ),
            "fontes": [],
            "rota": "documentos",
        }

    contexto = "\n\n---\n\n".join(
        f"[Fonte: {t['source']}]\n{t['texto']}" for t in trechos
    )

    resposta = cliente.models.generate_content(
        model=GERACAO_MODEL,
        contents=PROMPT_RESPOSTA_DOCUMENTOS.format(contexto=contexto, pergunta=pergunta),
        config=types.GenerateContentConfig(temperature=0.2),
    )

    fontes = sorted({t["source"] for t in trechos})
    return {"resposta": resposta.text, "fontes": fontes, "rota": "documentos"}


# ---------------------------------------------------------------------------
# 2b. Rota numérica (NL -> pandas -> execução restrita -> resposta)
# ---------------------------------------------------------------------------

PROMPT_GERAR_PANDAS = """Você tem acesso a DataFrames pandas carregados de uma planilha de
desempenho de uma empresa de lavanderia com 3 lojas (Matriz, Filial 1, Filial 2).
Os dados são brutos (cabeçalhos podem não estar na linha 0, valores podem ter texto
como "R$" ou "—" para ausência de dado) — considere isso ao escrever a expressão.

Esquema disponível (com o conteúdo completo das tabelas menores):
{esquema}

Regras importantes:
- NUNCA use .iloc com posição fixa adivinhada para localizar um mês ou nome — use
  filtragem por igualdade no valor da célula, ex.: df[df[0] == 'JULHO'][1].values[0]
  (o mês/nome pode estar em qualquer linha, a posição varia entre abas).
- Para pegar um valor único de uma célula já filtrada, use .values[0] ou .item()
  no final, não deixe uma Series inteira como resultado.
- Se o valor encontrado for um texto tipo "—" ou "-", trate como ausência de dado.

Escreva APENAS uma expressão Python válida (uma linha, sem explicação, sem markdown)
que calcule a resposta para a pergunta abaixo, usando os nomes de DataFrame exatamente
como aparecem no esquema (ex.: controle, referencia_pml, rafael, camila...).
Se não for possível responder com os dados disponíveis, responda exatamente: None

PERGUNTA: {pergunta}
"""

_BUILTINS_PERMITIDOS = {
    "len": len, "sum": sum, "round": round, "min": min, "max": max,
    "abs": abs, "sorted": sorted, "list": list, "dict": dict, "str": str,
    "int": int, "float": float, "range": range,
}


def _resultado_valido(resultado) -> bool:
    """
    Verifica se o resultado calculado é utilizável, e não um vazio disfarçado
    (NaN, None, string vazia, Series/DataFrame vazio ou só de NaN).
    Isso evita que o agente "formate" um vazio como se fosse uma resposta real.
    """
    if resultado is None:
        return False
    if isinstance(resultado, float) and pd.isna(resultado):
        return False
    if isinstance(resultado, (pd.Series, pd.DataFrame)):
        if resultado.empty:
            return False
        if resultado.isna().all().all() if isinstance(resultado, pd.DataFrame) else resultado.isna().all():
            return False
    if isinstance(resultado, str) and resultado.strip() in ("", "-", "—", "nan", "NaN"):
        return False
    return True


def _executar_consulta_numerica(cliente: genai.Client, pergunta: str):
    dataframes, esquema = _dataframes()

    resposta = cliente.models.generate_content(
        model=GERACAO_MODEL,
        contents=PROMPT_GERAR_PANDAS.format(esquema=esquema, pergunta=pergunta),
        config=types.GenerateContentConfig(temperature=0),
    )
    expressao = resposta.text.strip().strip("`").strip()

    if expressao == "None" or not expressao:
        return None, None

    namespace = {"pd": pd, **dataframes}
    try:
        resultado = eval(expressao, {"__builtins__": _BUILTINS_PERMITIDOS}, namespace)
        if not _resultado_valido(resultado):
            return None, expressao
        return resultado, expressao
    except Exception:
        return None, expressao


def _responder_via_numerico(cliente: genai.Client, pergunta: str) -> dict:
    resultado, expressao = _executar_consulta_numerica(cliente, pergunta)

    if resultado is None:
        return {
            "resposta": (
                "Não encontrei essa informação nos dados de desempenho disponíveis. "
                "Verifique se a pergunta se refere a um dado presente na planilha "
                "Av_Desempenho_Renove.xlsx."
            ),
            "fontes": [],
            "rota": "numerico",
        }

    resposta = cliente.models.generate_content(
        model=GERACAO_MODEL,
        contents=(
            f"Formate esta resposta de forma clara e direta para um colaborador, em "
            f"português, uma ou duas frases. Pergunta original: {pergunta!r}. "
            f"Resultado calculado: {resultado!r}"
        ),
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return {
        "resposta": resposta.text,
        "fontes": ["Av_Desempenho_Renove.xlsx"],
        "rota": "numerico",
        "_debug_expressao": expressao,
    }


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def responder_pergunta(pergunta: str) -> dict:
    cliente = _cliente()
    rota = _classificar_intencao(cliente, pergunta)

    if rota == "numerico":
        return _responder_via_numerico(cliente, pergunta)
    return _responder_via_documentos(cliente, pergunta)
