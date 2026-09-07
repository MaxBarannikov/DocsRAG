"""Optional API-key authentication, disabled while API_KEY is unset."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from core.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op unless API_KEY is configured."""
    if not settings.api_key:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
