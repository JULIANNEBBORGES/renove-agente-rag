"""
Vetorização dos chunks de documentos usando a API de embeddings do Gemini,
indexados num vector store local FAISS.

Usa o SDK atual "google-genai" (o pacote antigo "google-generativeai" foi
descontinuado pelo Google em 2026 — não usar).

Pré-requisito: rodar ingest.py antes (gera chunks.jsonl).
Pré-requisito: variável de ambiente GEMINI_API_KEY definida (.env local).

Uso:
    python vectorize.py

Saída:
    faiss_index.bin  — índice FAISS com os vetores de embedding
    faiss_meta.json  — lista paralela com o texto e metadados de cada vetor
                        (na mesma ordem em que foram inseridos no índice)
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.jsonl"
INDEX_FILE = BASE_DIR / "faiss_index.bin"
META_FILE = BASE_DIR / "faiss_meta.json"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
TASK_TYPE_DOC = "RETRIEVAL_DOCUMENT"

# Gemini tem rate limit no tier gratuito — pequena pausa entre chamadas evita erro 429.
PAUSA_ENTRE_CHAMADAS_SEG = 0.5


def criar_cliente() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY não encontrada. Crie um arquivo .env na raiz do projeto "
            "(veja .env.example) com sua chave do Google AI Studio."
        )
    return genai.Client(api_key=api_key)


def carregar_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} não encontrado. Rode 'python ingest.py' primeiro."
        )
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                chunks.append(json.loads(linha))
    return chunks


def gerar_embedding(cliente: genai.Client, texto: str) -> list:
    """Chama a API do Gemini para gerar o embedding de um texto."""
    resultado = cliente.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texto,
        config=types.EmbedContentConfig(
            task_type=TASK_TYPE_DOC,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return resultado.embeddings[0].values


def construir_indice(cliente: genai.Client, chunks):
    indice = faiss.IndexFlatIP(EMBEDDING_DIM)  # produto interno = similaridade de cosseno (vetores normalizados)
    metadados = []

    for i, chunk in enumerate(chunks):
        print(f"[{i + 1}/{len(chunks)}] Gerando embedding: {chunk['id']}")
        vetor = gerar_embedding(cliente, chunk["texto"])

        vetor_np = np.array(vetor, dtype="float32")
        faiss.normalize_L2(vetor_np.reshape(1, -1))
        indice.add(vetor_np.reshape(1, -1))

        metadados.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "categoria": chunk["categoria"],
            "texto": chunk["texto"],
        })

        time.sleep(PAUSA_ENTRE_CHAMADAS_SEG)

    return indice, metadados


def main():
    cliente = criar_cliente()
    chunks = carregar_chunks()
    print(f"Carregados {len(chunks)} chunks de {CHUNKS_FILE.name}\n")

    indice, metadados = construir_indice(cliente, chunks)

    faiss.write_index(indice, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    print(f"\nÍndice salvo em {INDEX_FILE} ({indice.ntotal} vetores)")
    print(f"Metadados salvos em {META_FILE}")


if __name__ == "__main__":
    main()
