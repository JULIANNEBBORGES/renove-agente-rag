"""
Orquestrador do agente RAG da Renove Lavanderias.

Escopo: agente de conhecimento sobre documentos internos (regras de assiduidade,
BPMN do processo de extração, tutorial de extração do Renov.net, guias técnicos
de treinamento). Busca semântica no índice FAISS (retrieval.py) + geração de
resposta com Gemini, sempre citando a fonte, restrita ao contexto recuperado.

NOTA DE ESCOPO: a consulta numérica à planilha de desempenho
(Av_Desempenho_Renove.xlsx) foi retirada do escopo do agente. A planilha tinha
problemas de qualidade de dados na origem (fórmulas com cache desatualizado,
estrutura irregular entre abas) que tornavam a extração de números confiável
demais arriscada pro prazo do desafio. Decisão consciente, documentada no README.

Se nada relevante for encontrado nos documentos, o agente informa isso
explicitamente em vez de arriscar uma resposta incorreta.

Usa o SDK atual "google-genai" (não "google-generativeai", descontinuado).

Uso (via Streamlit, ver app.py):
    from rag_agent import responder_pergunta
    resultado = responder_pergunta("Quantos atrasos de até 5 min são tolerados?")
"""

import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

import retrieval

load_dotenv()

GERACAO_MODEL = "gemini-3.5-flash-lite"

_cliente_cache = None


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


PROMPT_RESPOSTA_DOCUMENTOS = """Você é o agente interno de conhecimento da Renove Lavanderias.
Responda a pergunta do colaborador usando SOMENTE as informações do CONTEXTO abaixo.
Não use conhecimento externo. Se o contexto não for suficiente para responder com
segurança, diga claramente que não encontrou essa informação nos documentos disponíveis.

Ao final da resposta, cite a(s) fonte(s) usada(s) entre parênteses (nome do arquivo).

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}
"""


def responder_pergunta(pergunta: str) -> dict:
    cliente = _cliente()
    trechos = retrieval.buscar_trechos_relevantes(cliente, pergunta, k=4)

    if not trechos:
        return {
            "resposta": (
                "Não encontrei essa informação nos documentos disponíveis. "
                "Pode ser útil falar com a área responsável (RH ou gestão) pra confirmar."
            ),
            "fontes": [],
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
    return {"resposta": resposta.text, "fontes": fontes}
