"""End-to-end RAG pipeline: retrieve -> format context -> Anthropic generate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .vector_store import similarity_search


@dataclass
class RetrievedChunk:
    rank: int  # 1-indexed
    document: Document
    score: float

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def chunk_index(self) -> int:
        return int(self.document.metadata.get("chunk_index", 0))


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]
    model: str


def _format_context(chunks: Iterable[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        header = f"[{c.rank}] source={c.source} chunk={c.chunk_index} score={c.score:.3f}"
        blocks.append(f"{header}\n{c.document.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
    raw = similarity_search(query, k=k or settings.top_k)
    return [
        RetrievedChunk(rank=i + 1, document=doc, score=float(score))
        for i, (doc, score) in enumerate(raw)
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _call_anthropic(question: str, context: str) -> str:
    """Call Anthropic with retries on transient errors."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or Streamlit secrets."
        )
    # Imported lazily so the rest of the app remains usable without the SDK.
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_TEMPLATE.format(question=question, context=context),
            }
        ],
    )
    # `message.content` is a list of content blocks; concatenate text blocks.
    parts = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def answer_question(question: str, k: int | None = None) -> RagAnswer:
    """Retrieve relevant context and ask Claude for a cited answer."""
    question = question.strip()
    if not question:
        return RagAnswer(answer="Please enter a question.", sources=[], model=settings.anthropic_model)

    chunks = retrieve(question, k=k)
    if not chunks:
        return RagAnswer(
            answer=(
                "The knowledge base is empty or no relevant passages were found. "
                "Upload medical documents on the *Knowledge Base* page and try again."
            ),
            sources=[],
            model=settings.anthropic_model,
        )

    context = _format_context(chunks)
    try:
        answer = _call_anthropic(question, context)
    except Exception as exc:
        return RagAnswer(
            answer=(
                f"⚠️ The language model call failed: `{exc}`.\n\n"
                "Showing the retrieved passages below so you can still review the evidence."
            ),
            sources=chunks,
            model=settings.anthropic_model,
        )

    return RagAnswer(answer=answer, sources=chunks, model=settings.anthropic_model)
