"""Language routing for the RU-EN wrapper."""

from __future__ import annotations

import pytest

from api.translation import contains_cyrillic


@pytest.mark.parametrize(
    "text",
    [
        "Главные преимущества FastAPI?",
        "Что такое Depends?",
        "ёж",
        "Ёлка",
        "mixed English и русский",
    ],
)
def test_cyrillic_text_is_routed_through_translation(text: str) -> None:
    assert contains_cyrillic(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "How do I define a path parameter in FastAPI?",
        "",
        "   ",
        "@app.get('/items/{item_id}')",
        "naive resume",
    ],
)
def test_latin_text_skips_translation(text: str) -> None:
    assert contains_cyrillic(text) is False
