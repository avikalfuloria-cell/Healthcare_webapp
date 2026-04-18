"""CLI helper to ingest the bundled sample documents into the vector store.

Run from the project root:

    python -m scripts.ingest_samples
"""
from __future__ import annotations

from pathlib import Path

from src.document_loader import load_directory
from src.vector_store import add_documents, collection_stats


def main() -> None:
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    if not sample_dir.exists():
        raise SystemExit(f"Sample directory not found: {sample_dir}")

    print(f"Loading documents from {sample_dir} …")
    docs = load_directory(sample_dir)
    for d in docs:
        d.metadata.setdefault("doc_type", "sample")

    print(f"Embedding and indexing {len(docs)} chunks …")
    add_documents(docs)

    stats = collection_stats()
    print(
        f"Done. Collection `{stats['collection']}` now has "
        f"{stats['num_chunks']} chunks at {stats['persist_dir']}."
    )


if __name__ == "__main__":
    main()
