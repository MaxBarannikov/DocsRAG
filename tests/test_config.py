from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings


def test_pytorch_and_onnx_fp32_share_a_collection() -> None:
    """Parity-equivalent, so a separate FP32 index would be wasted work."""
    for backend in ("pytorch", "onnx-fp32"):
        settings = Settings.model_validate({"embedder_backend": backend, "qdrant_collection": "docsrag"})
        assert settings.active_qdrant_collection == "docsrag"


def test_int8_routes_to_its_own_collection() -> None:
    settings = Settings(embedder_backend="onnx-int8", qdrant_collection_int8="docsrag_int8")
    assert settings.active_qdrant_collection == "docsrag_int8"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedder_backend": "onnx-fp16"},
        {"inference_backend": "vllmm"},
        {"retrieval_strategy": "hybrd"},
    ],
    ids=["embedder_backend", "inference_backend", "retrieval_strategy"],
)
def test_unknown_backend_values_are_rejected(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(kwargs)


def test_assignment_is_validated() -> None:
    """The eval harness assigns embedder_backend at runtime."""
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.embedder_backend = "onnx-fp16"  # type: ignore[assignment]


def test_tracing_requires_both_langfuse_keys() -> None:
    assert Settings(langfuse_public_key="", langfuse_secret_key="").tracing_enabled is False
    assert Settings(langfuse_public_key="pk", langfuse_secret_key="").tracing_enabled is False
    assert Settings(langfuse_public_key="pk", langfuse_secret_key="sk").tracing_enabled is True  # noqa: S106


def test_bm25_cache_path_is_keyed_by_collection() -> None:
    """Otherwise switching backend reuses an index built from another corpus."""
    settings = Settings()
    assert settings.bm25_index_path("docsrag") != settings.bm25_index_path("docsrag_int8")


def test_invalid_numeric_ranges_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(top_k=0)
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=0)


def test_health_errors_name_the_remediation_step() -> None:
    """Errors must name the command that fixes them."""
    from unittest.mock import MagicMock

    from core.health import DependencyUnavailableError, check_qdrant

    unreachable = MagicMock()
    unreachable.collection_exists.side_effect = ConnectionError("[Errno 61] Connection refused")
    with pytest.raises(DependencyUnavailableError, match="make up"):
        check_qdrant(unreachable)

    empty = MagicMock()
    empty.collection_exists.return_value = False
    with pytest.raises(DependencyUnavailableError, match="make reindex"):
        check_qdrant(empty)

    no_points = MagicMock()
    no_points.collection_exists.return_value = True
    no_points.count.return_value.count = 0
    with pytest.raises(DependencyUnavailableError, match="make reindex"):
        check_qdrant(no_points)

    populated = MagicMock()
    populated.collection_exists.return_value = True
    populated.count.return_value.count = 2540
    assert check_qdrant(populated) == 2540


def test_mlflow_check_rejects_a_non_mlflow_responder() -> None:
    """macOS AirPlay Receiver holds port 5000 and answers 403 there."""
    from unittest.mock import MagicMock, patch

    from core.health import DependencyUnavailableError, check_mlflow

    response = MagicMock()
    response.status = 403
    response.__enter__.return_value = response
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(DependencyUnavailableError) as exc:
        check_mlflow()
    assert "403" in str(exc.value)
    assert "AirPlay" in str(exc.value)
