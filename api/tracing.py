"""LangFuse tracing, a no-op when unconfigured or unusable."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger

from core.config import settings

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler


@lru_cache(maxsize=1)
def get_langfuse_handler() -> BaseCallbackHandler | None:
    """Handler for this process, or None when tracing is off or the keys are rejected.

    Credentials are checked up front because the SDK exports spans in the background:
    bad keys otherwise surface as a bare 401 on stderr for every single request.
    """
    if not settings.tracing_enabled:
        return None

    # The SDK reads its host from the environment.
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        if not get_client().auth_check():
            logger.warning(
                "LangFuse credentials were rejected by {} — tracing is disabled for this run. "
                "Check the keys under Settings > API Keys, and that the project region "
                "matches LANGFUSE_HOST (the US and EU clouds issue different keypairs). "
                "Leave LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY empty to disable tracing "
                "deliberately.",
                settings.langfuse_host,
            )
            return None

        handler = CallbackHandler()
    except Exception as exc:  # noqa: BLE001 — tracing must never break a request
        logger.warning("LangFuse tracing is configured but the handler could not be created: {}", exc)
        return None

    logger.info("LangFuse tracing enabled (host={})", settings.langfuse_host)
    return handler
