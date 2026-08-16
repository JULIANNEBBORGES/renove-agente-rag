# Renove Lavanderias — Agente RAG (Challenge AluraAgente)

Repositório do desafio "Challenge AluraAgente" (ONE IA FOR TECH), com um agente de IA baseado em RAG
focado no **colaborador interno** da Renove Lavanderias Especializada (empresa fictícia) — não trata
de atendimento a cliente externo.

## Estrutura de pastas

```
repo_renove/
├── documentos/
│   └── gestao_processos_internos/   # Regras de negócio, BPMN, tutorial de extração
├── docs_treinamento/                # Guias técnicos (fibras, peles, segurança, etc.)
├── agente/
│   ├── ingest.py                    # Extrai e faz chunking dos documentos de texto
│   ├── vectorize.py                 # Gera embeddings (Gemini) e indexa no FAISS
│   ├── retrieval.py                 # Busca semântica no índice FAISS
│   └── rag_agent.py                 # Orquestra busca + geração de resposta
├── app.py                           # Interface de chat (Streamlit)
├── requirements.txt
└── .env.example
```

## Como rodar localmente

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
2. Copie `.env.example` para `.env` e cole sua chave do Gemini (gere em
   [aistudio.google.com](https://aistudio.google.com)).
3. Gere os chunks de texto dos documentos:
   ```
   cd agente && python ingest.py
   ```
4. Gere os embeddings e o índice vetorial (consome sua chave — ~27 chamadas):
   ```
   python vectorize.py
   ```
5. Volte pra raiz do projeto e rode o app:
   ```
   cd .. && streamlit run app.py
   ```

## Fontes de conhecimento do agente

| Fonte | Tipo de pergunta que responde |
|---|---|
| Regras de negócio + BPMN | Políticas e processos (ex.: regras de assiduidade) |
| Tutorial de extração | Como os dados são exportados/salvos (perguntas administrativas) |
| Guias de treinamento | Conhecimento técnico operacional (fibras, tecidos, segurança) |

> 📌 **Nota de escopo — planilha de desempenho**: a consulta numérica à planilha
> `Av_Desempenho_Renove.xlsx` foi retirada do escopo do agente. A planilha de origem
> apresentava problemas de qualidade de dados (fórmulas com cache desatualizado,
> estrutura irregular entre abas) que tornavam respostas numéricas confiáveis
> inviáveis dentro do prazo do desafio. Decisão consciente, priorizando um agente
> menor mas confiável em vez de um recurso adicional frágil.

> ⚠️ **Atenção — dados fictícios**: todos os nomes de colaboradores, clientes e dados operacionais
> neste repositório são fictícios, construídos a partir da estrutura de processos reais da empresa,
> mas sem qualquer informação pessoal real. Ver `documentos/gestao_processos_internos/Regras_Negocio_Renove.md`
> para pendências de conteúdo.

## Stack do agente

- **Embeddings + geração de resposta**: Google Gemini API (tier gratuito)
- **Vector store**: FAISS (local, embutido na aplicação)
- **Interface**: Streamlit
- **Hospedagem**: Streamlit Community Cloud, deploy direto a partir deste repositório

## Deploy

<img width="1906" height="990" alt="RENOV NET_AGENTE_1" src="https://github.com/user-attachments/assets/22a889f1-793f-4be5-98a7-2200d1b5c589" />
<img width="1709" height="994" alt="RENOV NET_AGENTE_2" src="https://github.com/user-attachments/assets/b44f8a20-7f35-4580-b0c3-b4c1ef8b4bc7" />

