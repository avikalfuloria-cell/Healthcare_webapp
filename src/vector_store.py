"""ChromaDB-backed vector store wrapper."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import settings
from .embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return a cached, on-disk Chroma collection."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def add_documents(docs: Iterable[Document]) -> list[str]:
    """Add LangChain documents to the collection. Returns the assigned IDs."""
    docs = list(docs)
    if not docs:
        return []
    store = get_vector_store()
    ids = store.add_documents(docs)
    # Chroma >=0.5 persists automatically; older versions need an explicit call.
    if hasattr(store, "persist"):
        try:
            store.persist()
        except Exception:
            pass
    return ids


def similarity_search(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Return (document, similarity_score) tuples for the top-k matches."""
    store = get_vector_store()
    return store.similarity_search_with_relevance_scores(query, k=k or settings.top_k)


def collection_stats() -> dict:
    """Return basic stats about the collection (for the UI)."""
    store = get_vector_store()
    try:
        count = store._collection.count()  # type: ignore[attr-defined]
    except Exception:
        count = None
    return {
        "collection": settings.chroma_collection,
        "persist_dir": str(settings.chroma_persist_dir),
        "embedding_model": settings.embedding_model,
        "num_chunks": count,
    }


def reset_collection() -> None:
    """Drop and recreate the collection. Used by the admin "clear" button."""
    store = get_vector_store()
    try:
        store.delete_collection()
    except Exception:
        pass
    # Bust the cache so the next call rebuilds it.
    get_vector_store.cache_clear()
