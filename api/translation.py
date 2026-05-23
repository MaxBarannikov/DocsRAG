"""RU↔EN translation for the RAG pipeline.

The corpus is English-only, so Russian questions are translated before retrieval
and answers are translated back. English questions bypass this entirely.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser

from api.prompts import TRANSLATE_EN_TO_RU_PROMPT, TRANSLATE_RU_TO_EN_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def contains_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def translate_to_english(llm: BaseChatModel, text: str, callbacks: list | None = None) -> str:
    chain = TRANSLATE_RU_TO_EN_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    return chain.invoke({"text": text}, config=config).strip()


def translate_to_russian(llm: BaseChatModel, text: str, callbacks: list | None = None) -> str:
    chain = TRANSLATE_EN_TO_RU_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    return chain.invoke({"text": text}, config=config).strip()
