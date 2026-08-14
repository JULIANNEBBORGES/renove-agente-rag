"""
Interface de chat do agente RAG interno da Renove Lavanderias.

Roda com:
    streamlit run app.py

Requer GEMINI_API_KEY definida em .env (local) ou em st.secrets (Streamlit Cloud).
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "agente"))

load_dotenv()

# No Streamlit Community Cloud, a chave fica em st.secrets, não em variável de ambiente.
# Copiamos pra os.environ pra manter o resto do código (rag_agent.py) igual em qualquer ambiente.
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

import rag_agent  # noqa: E402  (import depois do sys.path.insert de propósito)


st.set_page_config(page_title="Renov.net — Agente Interno", page_icon="🧺", layout="centered")

st.title("🧺 Renov.net — Agente de Conhecimento Interno")
st.caption(
    "Converse com um agente de IA (não uma pessoa) sobre regras de assiduidade, "
    "processos de extração de relatórios, guias técnicos e dados de desempenho "
    "da Renove Lavanderias. Respostas baseadas apenas nos documentos internos."
)

if "GEMINI_API_KEY" not in os.environ:
    st.error(
        "GEMINI_API_KEY não configurada. Crie um arquivo .env local (veja .env.example) "
        "ou configure em Settings → Secrets no Streamlit Community Cloud."
    )
    st.stop()

if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["conteudo"])
        if msg.get("fontes"):
            st.caption("📎 Fontes: " + ", ".join(msg["fontes"]))

pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    st.session_state.historico.append({"role": "user", "conteudo": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando os documentos internos..."):
            try:
                resultado = rag_agent.responder_pergunta(pergunta)
                resposta_texto = resultado["resposta"]
                fontes = resultado.get("fontes", [])
            except Exception as e:
                resposta_texto = (
                    "Ocorreu um erro ao consultar o agente. Tente novamente em instantes. "
                    f"(detalhe técnico: {e})"
                )
                fontes = []

        st.markdown(resposta_texto)
        if fontes:
            st.caption("📎 Fontes: " + ", ".join(fontes))

    st.session_state.historico.append(
        {"role": "assistant", "conteudo": resposta_texto, "fontes": fontes}
    )

with st.sidebar:
    st.subheader("Sobre este agente")
    st.markdown(
        "- 🤖 Você está falando com uma **IA**, não uma pessoa.\n"
        "- 📚 Fontes: regras de assiduidade, BPMN do processo mensal, tutorial de "
        "extração, guias técnicos de treinamento e a planilha de desempenho.\n"
        "- ⚠️ Se o agente não encontrar a informação, ele avisa em vez de arriscar "
        "uma resposta incorreta."
    )
    if st.button("🗑️ Limpar conversa"):
        st.session_state.historico = []
        st.rerun()
