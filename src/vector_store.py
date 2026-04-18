"""FAISS-backed vector store wrapper.

We use FAISS instead of ChromaDB because ChromaDB pulls in an
`opentelemetry-exporter-otlp-proto-grpc` dependency whose bundled protobuf
files are incompatible with the protobuf C++ implementation on
Streamlit Cloud. FAISS has no such baggage: just numpy + a small C++ index.

Persistence: we save the index to `<persist_dir>/faiss_index/` using
`FAISS.save_local`. On startup we try to load it; if it doesn't exist the
store starts empty.
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import settings
from .embeddings import get_embeddings


def _index_path():
    return settings.chroma_persist_dir / "faiss_index"


@lru_cache(maxsize=1)
def get_vector_store() -> FAISS | None:
    """Return the on-disk FAISS index, or None if the collection is empty."""
    path = _index_path()
    if not path.exists():
        return None
    return FAISS.load_local(
        str(path),
        get_embeddings(),
        allow_dangerous_deserialization=True,  # trust our own saved index
    )


def _persist(store: FAISS) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def add_documents(docs: Iterable[Document]) -> list[str]:
    """Add LangChain documents to the collection. Returns the assigned IDs."""
    docs = list(docs)
    if not docs:
        return []

    store = get_vector_store()
    if store is None:
        store = FAISS.from_documents(docs, get_embeddings())
        ids = [d.metadata.get("doc_id", "") for d in docs]
    else:
        ids = store.add_documents(docs)

    _persist(store)
    get_vector_store.cache_clear()  # next call reloads from disk
    return ids


def similarity_search(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Return (document, similarity_score) tuples for the top-k matches."""
    store = get_vector_store()
    if store is None:
        return []
    return store.similarity_search_with_relevance_scores(query, k=k or settings.top_k)


def collection_stats() -> dict:
    """Return basic stats about the collection (for the UI)."""
    store = get_vector_store()
    count: int | None
    if store is None:
        count = 0
    else:
        try:
            count = store.index.ntotal
        except Exception:
            count = None
    return {
        "collection": settings.chroma_collection,
        "persist_dir": str(settings.chroma_persist_dir),
        "embedding_model": settings.embedding_model,
        "num_chunks": count,
    }


def reset_collection() -> None:
    """Delete the on-disk FAISS index. Used by the admin "clear" button."""
    path = _index_path()
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    get_vector_store.cache_clear()
