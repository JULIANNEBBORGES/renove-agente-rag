"""
Camada de recuperação (retrieval) — busca semântica no índice FAISS.

Carrega o índice e os metadados gerados por vectorize.py, transforma a pergunta
do colaborador em embedding (mesmo modelo usado na indexação) e retorna os
trechos de documentos mais relevantes semanticamente.

Usa o SDK atual "google-genai".
"""

import json
from pathlib import Path

import faiss
import numpy as np
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "faiss_index.bin"
META_FILE = BASE_DIR / "faiss_meta.json"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# Limiar mínimo de similaridade (cosseno) para considerar um trecho relevante.
# Abaixo disso, o agente prefere dizer "não encontrei" a arriscar uma resposta ruim.
LIMIAR_RELEVANCIA = 0.55


class IndiceNaoEncontrado(Exception):
    pass


def _carregar_indice_e_metadados():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise IndiceNaoEncontrado(
            "Índice vetorial não encontrado. Rode 'python vectorize.py' primeiro "
            "(depois de rodar 'python ingest.py')."
        )
    indice = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadados = json.load(f)
    return indice, metadados


def _embedding_da_pergunta(cliente: genai.Client, pergunta: str) -> np.ndarray:
    resultado = cliente.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=pergunta,
        config=types.EmbedContentConfig(
            task_type=TASK_TYPE_QUERY,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    vetor = np.array(resultado.embeddings[0].values, dtype="float32")
    faiss.normalize_L2(vetor.reshape(1, -1))
    return vetor.reshape(1, -1)


def buscar_trechos_relevantes(cliente: genai.Client, pergunta: str, k: int = 4):
    """
    Retorna até k trechos mais relevantes pra pergunta, já filtrados pelo
    limiar de relevância. Cada item: {"texto", "source", "categoria", "score"}.
    """
    indice, metadados = _carregar_indice_e_metadados()
    vetor_pergunta = _embedding_da_pergunta(cliente, pergunta)

    scores, indices = indice.search(vetor_pergunta, k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < LIMIAR_RELEVANCIA:
            continue
        meta = metadados[idx]
        resultados.append({
            "texto": meta["texto"],
            "source": meta["source"],
            "categoria": meta["categoria"],
            "score": float(score),
        })

    return resultados
