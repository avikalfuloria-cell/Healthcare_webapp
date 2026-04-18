"""About / architecture / safety page."""
import streamlit as st

from src.theme import render_theme_toggle

st.set_page_config(page_title="About · MedLit-RAG", page_icon="ℹ️", layout="wide")
render_theme_toggle()

st.title("ℹ️ About MedLit-RAG")

st.markdown(
    """
**MedLit-RAG** is a retrieval-augmented question-answering app over a corpus
of medical literature you control. It is built for **information lookup and
literature synthesis** — *not* for clinical decision-making.

## How it works

1. **Ingest.** Documents (PDF / DOCX / TXT / MD) are split into ~800-token
   chunks with overlap, preserving paragraph and sentence boundaries.
2. **Embed.** Each chunk is embedded with
   [`pritamdeka/S-PubMedBert-MS-MARCO`](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO),
   a PubMedBERT model fine-tuned on MS MARCO for retrieval. Biomedical
   vocabulary, drug names, and gene symbols are handled natively.
3. **Store.** Embeddings + metadata are persisted in a local **ChromaDB**
   collection at `./data/chroma/`.
4. **Retrieve.** At query time we run a top-k similarity search against the
   collection and rank passages by relevance score.
5. **Generate.** The retrieved passages are formatted as numbered sources
   and passed, with the user's question, to **Anthropic Claude**. The system
   prompt forces the model to cite every claim and to refuse if the sources
   are insufficient.

## Tech stack

| Layer        | Choice                                       |
|--------------|----------------------------------------------|
| LLM          | Anthropic Claude (`claude-sonnet-4-6`)       |
| Orchestration| LangChain                                    |
| Embeddings   | PubMedBERT (S-PubMedBert-MS-MARCO)           |
| Vector store | ChromaDB (on-disk, persistent)               |
| UI           | Streamlit                                    |

## Safety & limitations

- **Not a medical device.** Outputs are not a substitute for professional
  clinical judgment.
- **Garbage in, garbage out.** Answers are only as good as the documents
  you've ingested. Curate your corpus.
- **No PHI.** Do not upload identifiable patient data unless your deployment
  is HIPAA-compliant and your LLM provider has signed a BAA. Streamlit
  Community Cloud is *not* HIPAA-compliant.
- **Citations may still be wrong.** Always verify by clicking through to
  the source passage shown under each answer.
- **Hallucination guardrails.** The system prompt instructs the model to
  refuse when the context is insufficient, but no guardrail is perfect.

## Project layout

```
healthcare-rag-app/
├── app.py                      # Chat / Q&A page
├── pages/
│   ├── 1_📚_Knowledge_Base.py  # Upload, inspect, reset
│   └── 2_ℹ️_About.py           # This page
├── src/
│   ├── config.py               # Settings (env + Streamlit secrets)
│   ├── embeddings.py           # PubMedBERT embeddings
│   ├── vector_store.py         # Chroma wrapper
│   ├── document_loader.py      # PDF/DOCX/TXT → chunked Documents
│   ├── prompts.py              # System + user prompts
│   └── rag_chain.py            # retrieve → format → Claude
├── data/
│   ├── sample/                 # Bundled sample documents
│   ├── chroma/                 # Chroma persistent index (gitignored)
│   └── uploads/                # Saved user uploads (gitignored)
├── scripts/
│   └── ingest_samples.py       # CLI ingest
├── requirements.txt
├── .env.example
└── README.md
```
"""
)
