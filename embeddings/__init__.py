"""Pluggable embedder backends.

Only the factory is re-exported: each backend imports heavy dependencies at module
load, so `make_embedder()` pulls in exactly one of them. Import the concrete classes
from their own modules when you need the type.
"""

from embeddings.factory import make_embedder

__all__ = ["make_embedder"]
