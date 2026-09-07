"""Preflight checks so the CLIs fail with an actionable message, not a driver traceback."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class DependencyUnavailableError(RuntimeError):
    """Carries a message meant for a human, including the command that fixes it."""


def check_qdrant(client: QdrantClient, *, require_collection: bool = True) -> int:
    """Return the point count of the active collection, or raise with remediation steps."""
    collection = settings.active_qdrant_collection

    try:
        exists = client.collection_exists(collection)
    except Exception as exc:
        msg = f"Qdrant is not reachable at {settings.qdrant_url} ({exc}).\nStart the stack with `make up`, then retry."
        raise DependencyUnavailableError(msg) from exc

    if not exists:
        if not require_collection:
            return 0
        msg = (
            f"Qdrant is running, but collection {collection!r} does not exist.\n"
            f"Build the index with `make reindex`, then retry."
        )
        raise DependencyUnavailableError(msg)

    count = int(client.count(collection, exact=True).count)
    if require_collection and count == 0:
        msg = f"Collection {collection!r} exists but is empty.\nBuild the index with `make reindex`, then retry."
        raise DependencyUnavailableError(msg)
    return count


def check_mlflow() -> None:
    """Fail before a long run rather than after it, when results would be lost."""
    import urllib.error
    import urllib.request

    url = f"{settings.mlflow_tracking_uri.rstrip('/')}/api/2.0/mlflow/experiments/search?max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 — URL from our own settings
            if response.status == HTTPStatus.OK:
                return
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, OSError) as exc:
        msg = (
            f"MLflow is not reachable at {settings.mlflow_tracking_uri} ({exc}).\n"
            f"Start the stack with `make up`, then retry."
        )
        raise DependencyUnavailableError(msg) from exc

    msg = (
        f"{settings.mlflow_tracking_uri} answered HTTP {status}, which is not MLflow.\n"
        f"On macOS this is usually AirPlay Receiver holding the port — disable it in\n"
        f"System Settings > General > AirDrop & Handoff, or set MLFLOW_TRACKING_URI to another port."
    )
    raise DependencyUnavailableError(msg)


def check_ollama() -> None:
    if settings.inference_backend != "ollama":
        return

    import urllib.error
    import urllib.request

    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5):  # noqa: S310 — URL comes from our own settings
            return
    except (urllib.error.URLError, OSError) as exc:
        msg = (
            f"Ollama is not responding at {settings.ollama_base_url} ({exc}).\n"
            f"Start the Ollama app, or run `ollama serve`, then retry."
        )
        raise DependencyUnavailableError(msg) from exc
