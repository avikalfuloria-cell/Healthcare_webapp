"""MedLit-RAG: Streamlit entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import settings
from src.rag_chain import answer_question
from src.theme import render_theme_toggle
from src.vector_store import collection_stats

st.set_page_config(
    page_title="MedLit-RAG · Medical Literature Q&A",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_theme_toggle()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 MedLit-RAG")
    st.caption("Retrieval-Augmented Q&A over your medical literature.")

    stats = collection_stats()
    st.markdown("### Knowledge base")
    st.metric("Indexed chunks", stats["num_chunks"] if stats["num_chunks"] is not None else "—")
    st.caption(f"Collection: `{stats['collection']}`")
    st.caption(f"Embedding model: `{stats['embedding_model']}`")

    st.markdown("### Generation")
    if settings.is_llm_configured:
        st.success(f"Anthropic ready · `{settings.anthropic_model}`")
    else:
        st.error("`ANTHROPIC_API_KEY` not set. Add it to `.env` or Streamlit secrets.")

    top_k = st.slider("Passages to retrieve (k)", 1, 12, settings.top_k)

    st.markdown("---")
    st.caption(
        "⚠️ **Not medical advice.** This tool surfaces information from documents "
        "you've ingested. Always consult a qualified clinician for personal "
        "medical decisions."
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Ask the medical literature")
st.write(
    "Type a clinical question. The app retrieves the most relevant passages "
    "from your knowledge base and asks Claude to answer with inline citations."
)

# Persist chat across reruns.
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(question, answer, sources)]

example_questions = [
    "What is the recommended first-line treatment for type 2 diabetes?",
    "Summarize the evidence for SGLT2 inhibitors in heart failure.",
    "What are the contraindications for metformin?",
]
with st.expander("Example questions", expanded=False):
    cols = st.columns(len(example_questions))
    for col, q in zip(cols, example_questions):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q

question = st.chat_input("Ask a medical-literature question…")
if "pending_question" in st.session_state and not question:
    question = st.session_state.pop("pending_question")

if question:
    with st.spinner("Retrieving passages and asking Claude…"):
        result = answer_question(question, k=top_k)
    st.session_state.history.append(
        {"question": question, "answer": result.answer, "sources": result.sources, "model": result.model}
    )

# Render history (newest first).
for turn in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for c in turn["sources"]:
                    st.markdown(
                        f"**[{c.rank}] {c.source}** · chunk {c.chunk_index} · "
                        f"score `{c.score:.3f}`"
                    )
                    st.caption(c.document.page_content[:600] + ("…" if len(c.document.page_content) > 600 else ""))
        st.caption(f"Model: `{turn['model']}`")
