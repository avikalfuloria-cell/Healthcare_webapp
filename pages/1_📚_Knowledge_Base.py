"""Manage the medical document knowledge base: upload, inspect, reset."""
from __future__ import annotations

# Force pure-Python protobuf implementation BEFORE chromadb is imported
# (via src.vector_store). See app.py for context.
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from pathlib import Path

import streamlit as st

from src.config import settings
from src.document_loader import load_bytes, load_directory
from src.theme import render_theme_toggle
from src.vector_store import add_documents, collection_stats, reset_collection

st.set_page_config(page_title="Knowledge Base · MedLit-RAG", page_icon="📚", layout="wide")
render_theme_toggle()

st.title("📚 Knowledge Base")
st.write(
    "Upload medical journal articles, clinical guidelines, drug monographs, "
    "or other reference documents. Files are chunked, embedded with a "
    "PubMedBERT-based model, and stored in a local ChromaDB collection."
)

stats = collection_stats()
col1, col2, col3 = st.columns(3)
col1.metric("Indexed chunks", stats["num_chunks"] if stats["num_chunks"] is not None else "—")
col2.metric("Collection", stats["collection"])
col3.metric("Embedding model", stats["embedding_model"].split("/")[-1])

st.markdown("---")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.subheader("Upload documents")
st.caption("Supported formats: PDF, DOCX, TXT, MD. Max 50 MB per file.")

uploaded = st.file_uploader(
    "Drop files here",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

doc_type = st.selectbox(
    "Tag these documents as…",
    options=["journal_article", "clinical_guideline", "drug_monograph", "review", "other"],
    index=0,
)

if uploaded and st.button("Ingest into knowledge base", type="primary"):
    total_chunks = 0
    progress = st.progress(0.0)
    for i, f in enumerate(uploaded, start=1):
        try:
            chunks = load_bytes(
                f.name,
                f.getvalue(),
                extra_metadata={"doc_type": doc_type},
            )
            add_documents(chunks)
            total_chunks += len(chunks)
            # Save a copy under data/uploads so the file persists across restarts.
            (settings.upload_dir / f.name).write_bytes(f.getvalue())
            st.write(f"✅ `{f.name}` → {len(chunks)} chunks")
        except Exception as exc:
            st.error(f"❌ `{f.name}` failed: {exc}")
        progress.progress(i / len(uploaded))
    st.success(f"Indexed {total_chunks} new chunks across {len(uploaded)} files.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Bulk-load samples
# ---------------------------------------------------------------------------
st.subheader("Load bundled sample documents")
sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
st.caption(f"Source directory: `{sample_dir}`")

if st.button("Ingest sample documents"):
    if not sample_dir.exists():
        st.error("Sample directory not found.")
    else:
        chunks = load_directory(sample_dir)
        for c in chunks:
            c.metadata.setdefault("doc_type", "sample")
        add_documents(chunks)
        st.success(f"Indexed {len(chunks)} chunks from sample documents.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Danger zone
# ---------------------------------------------------------------------------
st.subheader("Danger zone")
with st.expander("Clear the entire knowledge base"):
    st.warning(
        "This permanently deletes the Chroma collection. "
        "Files saved to `data/uploads/` are not removed."
    )
    confirm = st.text_input("Type CLEAR to confirm:")
    if st.button("Delete collection", disabled=(confirm != "CLEAR")):
        reset_collection()
        st.success("Collection cleared. Refresh to see the empty state.")
