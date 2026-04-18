"""Centralized configuration loaded from environment or Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env if present (no-op when running on Streamlit Cloud).
load_dotenv()


def _from_streamlit_secrets(key: str) -> str | None:
    """Read a value from `st.secrets` without importing streamlit at module import time."""
    try:
        import streamlit as st  # local import keeps this module importable in scripts
    except Exception:
        return None
    try:
        if key in st.secrets:
            value = st.secrets[key]
            return str(value) if value is not None else None
    except Exception:
        # st.secrets raises if no secrets file exists; treat as missing.
        return None
    return None


def _get(key: str, default: Any = None) -> Any:
    """Resolve config key from env first, then Streamlit secrets, then default."""
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    val = _from_streamlit_secrets(key)
    if val is not None and val != "":
        return val
    return default


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    anthropic_model: str
    embedding_model: str
    chroma_persist_dir: Path
    chroma_collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    upload_dir: Path

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    persist = Path(_get("CHROMA_PERSIST_DIR", project_root / "data" / "chroma"))
    upload = Path(_get("UPLOAD_DIR", project_root / "data" / "uploads"))
    persist.mkdir(parents=True, exist_ok=True)
    upload.mkdir(parents=True, exist_ok=True)

    return Settings(
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
        anthropic_model=_get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        embedding_model=_get("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"),
        chroma_persist_dir=persist,
        chroma_collection=_get("CHROMA_COLLECTION", "medical_literature"),
        chunk_size=int(_get("CHUNK_SIZE", 800)),
        chunk_overlap=int(_get("CHUNK_OVERLAP", 120)),
        top_k=int(_get("TOP_K", 5)),
        upload_dir=upload,
    )


settings = load_settings()
