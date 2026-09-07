from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class Embedder(Protocol):
    """Contract for embedder backends.

    Vectors must be L2-normalized: the collections use cosine distance, and
    un-normalized vectors degrade retrieval silently rather than raising.
    """

    model_name: str
    dimension: int

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress: bool = True,
        prefix: str = "",
    ) -> list[list[float]]: ...
