"""RU-EN wrapper over the English-only index. English questions skip both hops."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser

from api.prompts import TRANSLATE_EN_TO_RU_PROMPT, TRANSLATE_RU_TO_EN_PROMPT

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableConfig

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def contains_cyrillic(text: str) -> bool:
    """Routing signal for translation.

    Deliberately crude: also fires on other Cyrillic languages, misses Latin
    transliteration. The worst case is a needless translation hop.
    """
    return bool(_CYRILLIC_RE.search(text))


def _translate(
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
    text: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> str:
    chain = prompt | llm | StrOutputParser()
    config: RunnableConfig = {"callbacks": callbacks} if callbacks else {}
    return chain.invoke({"text": text}, config=config).strip()


def translate_to_english(
    llm: BaseChatModel,
    text: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> str:
    return _translate(TRANSLATE_RU_TO_EN_PROMPT, llm, text, callbacks)


def translate_to_russian(
    llm: BaseChatModel,
    text: str,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> str:
    return _translate(TRANSLATE_EN_TO_RU_PROMPT, llm, text, callbacks)
