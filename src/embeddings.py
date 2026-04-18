"""Domain-tuned embeddings for medical text.

We use `sentence-transformers` with a PubMedBERT-based model fine-tuned on
MS MARCO for retrieval. This handles biomedical vocabulary, abbreviations,
and gene/drug names better than a general-purpose model.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached LangChain-compatible embeddings object."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Convenience helper for direct embedding calls (e.g. ad-hoc scripts)."""
    return get_embeddings().embed_documents(list(texts))
