"""Runtime light/dark theme toggle for Streamlit.

Streamlit's `[theme]` block in `config.toml` is applied at server startup
and cannot be mutated from Python at runtime. To support an in-app toggle
we inject a small CSS override that flips the page's color variables when
the user picks "Light" in the sidebar. The default theme (in
`.streamlit/config.toml`) is dark.
"""
from __future__ import annotations

import streamlit as st

_LIGHT_CSS = """
<style id="medlit-light-theme">
  :root, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    --background-color: #FFFFFF;
    --secondary-background-color: #F2F6F5;
    --text-color: #0B1F2A;
  }
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    color: #0B1F2A !important;
  }
  [data-testid="stSidebar"] {
    background-color: #F2F6F5 !important;
  }
  [data-testid="stSidebar"] * ,
  [data-testid="stMain"] *,
  [data-testid="stMarkdownContainer"] * {
    color: #0B1F2A !important;
  }
  [data-testid="stChatMessage"] {
    background-color: #F2F6F5 !important;
  }
  /* Inputs */
  textarea, input, select, [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #0B1F2A !important;
  }
  /* Code blocks stay readable */
  code, pre, kbd {
    background-color: #EEF2F1 !important;
    color: #0B1F2A !important;
  }
  /* Buttons keep brand color */
  .stButton > button {
    background-color: #0E7C7B !important;
    color: #FFFFFF !important;
    border: 1px solid #0E7C7B !important;
  }
</style>
"""


def render_theme_toggle(default: str = "Dark") -> str:
    """Render the toggle in the sidebar and inject CSS if Light is selected.

    Returns the active theme name ("Dark" or "Light") so callers can branch
    on it if they need to (e.g. swap a logo).
    """
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = default

    with st.sidebar:
        st.session_state.theme_mode = st.radio(
            "Appearance",
            options=["Dark", "Light"],
            index=0 if st.session_state.theme_mode == "Dark" else 1,
            horizontal=True,
            key="_theme_radio",
        )

    if st.session_state.theme_mode == "Light":
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

    return st.session_state.theme_mode
