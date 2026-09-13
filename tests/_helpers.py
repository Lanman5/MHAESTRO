"""Shared test helpers.

`mhaestro.config.get_secret` reads Streamlit secrets before environment
variables, so a real `.streamlit/secrets.toml` on the developer's machine
silently overrides anything a test sets in `os.environ`. A test that hardcodes
`RESEARCHER_PIN = "test-pin"` therefore passes on a clean checkout and fails the
moment the study is actually configured -- which is the worst possible time for
the suite to start lying.

These helpers resolve a secret the same way the app does, so a test drives the
app with whatever value is genuinely in force.
"""
from __future__ import annotations

import os


def effective_secret(name: str, fallback: str) -> str:
    """The value the app will actually see for `name`."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No secrets file, or no Streamlit context -- fall through to the env.
        pass
    return os.environ.get(name, fallback)
