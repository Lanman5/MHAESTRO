"""Secret and configuration resolution.

Values are looked up in Streamlit secrets first (how the apps run on Streamlit
Community Cloud) and fall back to environment variables (how they run locally).
Nothing here raises on a missing key -- callers decide whether a key is required,
so a session can still run on a single provider.
"""
from __future__ import annotations

import os
from typing import Any, Optional


def get_secret(name: str, default: Optional[Any] = None) -> Optional[Any]:
    """Return a secret from st.secrets, then os.environ, then `default`."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # No Streamlit context, or no secrets.toml -- fall through to env vars.
        pass
    return os.environ.get(name, default)


def has_secret(name: str) -> bool:
    value = get_secret(name)
    return value is not None and str(value).strip() != ""
