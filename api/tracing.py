"""LangFuse tracing helpers — no-ops when keys are not configured."""

from __future__ import annotations

import os
from typing import Any

# LangFuse uses httpx internally — unset SOCKS proxy vars to avoid import errors.
for _var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_var, None)

from api.config import settings  # noqa: E402 — import after proxy cleanup


def get_langfuse_handler(question: str = "") -> Any | None:
    """Return a LangFuse CallbackHandler for this request, or None if tracing is not configured."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        return None
