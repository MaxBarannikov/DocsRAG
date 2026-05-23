from __future__ import annotations

from typing import TYPE_CHECKING

from api.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# Explicit cap: vllm-metal can truncate long answers; Ollama defaults to unlimited (num_predict=-1).
MAX_TOKENS = 1024


def make_llm(temperature: float = 0.0, *, json_mode: bool = False) -> BaseChatModel:
    if settings.inference_backend == "vllm":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key="EMPTY",  # vLLM ignores the key but langchain requires it
            temperature=temperature,
            top_p=1.0,
            max_tokens=MAX_TOKENS,
            frequency_penalty=0.3,
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        )

    from langchain_ollama import ChatOllama  # noqa: PLC0415

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        top_p=1.0,
        num_predict=MAX_TOKENS,  # Ollama's name for max_tokens
        format="json" if json_mode else "",
    )
