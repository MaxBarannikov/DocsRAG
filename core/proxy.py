from __future__ import annotations

import os

from loguru import logger

_PROXY_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


def strip_socks_proxy_env() -> list[str]:
    """Drop proxy variables so httpx clients can be constructed, returning what was removed.

    httpx raises at construction time when ALL_PROXY is a socks5:// URL and the socksio
    extra is missing, which breaks langchain-ollama, langfuse and ragas alike. Mutating
    global state is why this is called explicitly from entry points rather than on import.
    """
    removed = [name for name in _PROXY_VARS if os.environ.pop(name, None) is not None]
    if removed:
        logger.debug("Removed proxy env vars: {}", removed)
    return removed
