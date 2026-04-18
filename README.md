# 🩺 MedLit-RAG — Healthcare Literature Q&A

A Retrieval-Augmented Generation (RAG) web app for **medical literature
question answering**. Upload journal articles, clinical guidelines, or drug
monographs; ask questions in natural language; get answers grounded in your
documents with **inline citations**.

| Layer        | Choice                                                    |
|--------------|-----------------------------------------------------------|
| LLM          | Anthropic Claude (`claude-sonnet-4-6`)                    |
| Orchestration| LangChain                                                 |
| Embeddings   | PubMedBERT (`pritamdeka/S-PubMedBert-MS-MARCO`) on CPU    |
| Vector store | ChromaDB (local, on-disk, persistent)                     |
| UI           | Streamlit (multi-page)                                    |

> ⚠️ **Not a medical device.** This tool is for **information lookup and
> literature synthesis** only. It must not be used for diagnosis or
> individualized treatment decisions, and must not be fed identifiable
> patient data unless deployed in a HIPAA-compliant environment with a BAA
> in place with your LLM provider.

---

## Features

- 💬 **Chat-style Q&A** over your medical knowledge base.
- 📚 **Document upload** (PDF, DOCX, TXT, MD) with PubMedBERT embeddings.
- 🔎 **Citation-aware retrieval** — every answer shows the source passages
  and ranks them by relevance score.
- 🛡️ **Conservative system prompt** — Claude refuses if the retrieved
  context is insufficient, never invents citations, and declines to give
  individualized clinical advice.
- 🧰 **Knowledge-base management page** — inspect stats, ingest bundled
  samples, or wipe and rebuild the collection.
- 🚀 **Deployable on Streamlit Community Cloud** with `secrets.toml`
  configuration.

---

## Project layout

```
healthcare-rag-app/
├── app.py                      # Chat / Q&A page (Streamlit entry)
├── pages/
│   ├── 1_📚_Knowledge_Base.py  # Upload, inspect, reset collection
│   └── 2_ℹ️_About.py           # Architecture + safety
├── src/
│   ├── config.py               # Settings (env + Streamlit secrets)
│   ├── embeddings.py           # PubMedBERT embeddings via sentence-transformers
│   ├── vector_store.py         # ChromaDB wrapper
│   ├── document_loader.py      # PDF / DOCX / TXT → chunked LangChain Documents
│   ├── prompts.py              # System + user prompts
│   └── rag_chain.py            # retrieve → format → Anthropic generate
├── data/
│   ├── sample/                 # Bundled educational sample documents
│   ├── chroma/                 # Persistent Chroma index (gitignored)
│   └── uploads/                # Saved user uploads (gitignored)
├── scripts/
│   └── ingest_samples.py       # CLI: populate vector store from data/sample
├── .streamlit/
│   ├── config.toml             # Theme + server config
│   └── secrets.toml.example    # Template for Streamlit Cloud secrets
├── .github/workflows/ci.yml    # Lint + smoke-import CI
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 1. Local setup

### Prerequisites

- Python **3.10+** (3.11 recommended).
- ~2 GB free disk for the PubMedBERT model + Chroma index.
- An Anthropic API key — get one at <https://console.anthropic.com/>.

### Install

```bash
git clone https://github.com/<your-org>/healthcare-rag-app.git
cd healthcare-rag-app

python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

### Populate the knowledge base with bundled samples (optional)

```bash
python -m scripts.ingest_samples
```

This embeds the documents in `data/sample/` (metformin, SGLT2 inhibitors in
heart failure, hypertension management) into the local Chroma collection.

### Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>. Use the **📚 Knowledge Base** page to upload
your own PDFs or DOCX files, then return to the chat to ask questions.

---

## 2. Deploying on Streamlit Community Cloud

1. **Push to GitHub.** Create a public or private repo containing this
   project (do not commit `.env` or `data/chroma/`).
2. **Create a new app** on <https://share.streamlit.io/> and point it at
   your repo + branch + `app.py`.
3. **Add secrets.** In *App settings → Secrets*, paste the contents of
   [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) with
   your real `ANTHROPIC_API_KEY`.
4. **Deploy.** First boot will take a few minutes because the PubMedBERT
   model (~440 MB) downloads on cold start.

### Notes on Streamlit Cloud

- **Persistence is ephemeral.** The container's filesystem resets on
  redeploy, so the Chroma index in `data/chroma/` will be lost. For a
  persistent collection, mount a remote vector DB (e.g. swap Chroma for
  Pinecone) or use a paid hosting tier with persistent volumes.
- **Not HIPAA-compliant.** Do not upload identifiable patient data.
- **Cold starts** are slow because of the embedding-model download; the
  first request after a deploy may take 60–120 s.

### Alternative deployments

- **Hugging Face Spaces** (Streamlit SDK) — same flow, set `ANTHROPIC_API_KEY`
  as a Space secret.
- **Docker / Cloud Run / Fly.io** — write a small Dockerfile based on
  `python:3.11-slim`, copy the project, `pip install -r requirements.txt`,
  and run `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.
- **On-prem** for sensitive data — host on a server inside your trust
  boundary and either swap Anthropic for a self-hosted LLM (e.g. via vLLM)
  or use a HIPAA-eligible LLM provider with a signed BAA.

---

## 3. How the RAG pipeline works

```
    user question
         │
         ▼
 ┌──────────────────┐     embed (PubMedBERT)     ┌─────────────────┐
 │  retriever       │ ─────────────────────────► │  ChromaDB       │
 │  (top-k cosine)  │ ◄───── top-k chunks ────── │  (persistent)   │
 └─────────┬────────┘                             └─────────────────┘
           │ numbered, scored passages
           ▼
 ┌──────────────────┐
 │  prompt builder  │  → system prompt forces "answer only from sources"
 └─────────┬────────┘    + inline [#] citations
           ▼
 ┌──────────────────┐
 │  Anthropic       │  → claude-sonnet-4-6
 │  Claude          │
 └─────────┬────────┘
           ▼
   cited answer + source previews
```

The prompt in [src/prompts.py](src/prompts.py) is the most important
guardrail — read and tune it for your use case.

---

## 4. Configuration reference

All settings can be supplied via `.env` (local) or `.streamlit/secrets.toml`
(Streamlit Cloud). Environment variables win.

| Key                  | Default                                  | Purpose                           |
|----------------------|------------------------------------------|-----------------------------------|
| `ANTHROPIC_API_KEY`  | —                                        | Required for the generation step. |
| `ANTHROPIC_MODEL`    | `claude-sonnet-4-6`                      | Generation model.                 |
| `EMBEDDING_MODEL`    | `pritamdeka/S-PubMedBert-MS-MARCO`       | sentence-transformers model id.   |
| `CHROMA_PERSIST_DIR` | `./data/chroma`                          | On-disk Chroma path.              |
| `CHROMA_COLLECTION`  | `medical_literature`                     | Chroma collection name.           |
| `CHUNK_SIZE`         | `800`                                    | Characters per chunk.             |
| `CHUNK_OVERLAP`      | `120`                                    | Overlap between adjacent chunks.  |
| `TOP_K`              | `5`                                      | Default retrieval depth.          |

---

## 5. Security & responsible-use checklist

Before pointing real users at this app:

- [ ] Confirm the LLM provider has a signed BAA with you if any uploaded
      content can contain PHI. **Anthropic's standard API is not BAA-covered
      by default** — check current terms.
- [ ] Deploy behind authentication (e.g. SSO via a reverse proxy, or
      Streamlit Cloud's "Private app" sharing).
- [ ] Lock down file uploads — set `server.maxUploadSize` to a sensible
      cap and consider scanning uploads for PII before ingesting.
- [ ] Add rate limiting at the proxy layer to prevent API-cost abuse.
- [ ] Show the disclaimer prominently — the app already does this in the
      sidebar; do not remove it.
- [ ] Log queries and retrieved sources (with consent) so you can audit
      hallucinations and refine the corpus.
- [ ] Periodically re-embed when you change the embedding model.

---

## 6. Extending

- **Swap the vector DB.** Replace `src/vector_store.py` with a
  `langchain_community.vectorstores.Pinecone` / `Weaviate` / `FAISS`
  implementation; everything else stays the same.
- **Swap the embedding model.** Try `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
  for biomedical NLI, or `BAAI/bge-m3` for a general high-quality model.
  Re-embed after switching (use the *Clear collection* button + re-ingest).
- **Swap the LLM.** `src/rag_chain.py::_call_anthropic` is the only LLM-
  specific code — replace with an OpenAI / vLLM / Bedrock client.
- **Add hybrid search.** Combine BM25 (`rank_bm25`) with the dense
  retriever for better recall on rare drug names and acronyms.
- **Add reranking.** Pass the top-k passages through a cross-encoder
  (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) before sending to the LLM.

---

## 7. License

MIT for the application code in this repository. The bundled sample
documents under `data/sample/` are original synthetic summaries written for
demo purposes and are released under CC0 — replace them with your own
properly-licensed corpus before deployment.
