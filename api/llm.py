"""Chat model factory for the configured inference backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

from core.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def make_llm(temperature: float = 0.0, *, json_mode: bool = False) -> BaseChatModel:
    """Sampling parameters are explicit because Ollama and vLLM defaults differ,
    which would otherwise make any side-by-side benchmark unfair.
    """
    if settings.inference_backend == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key=SecretStr(settings.vllm_api_key),
            temperature=temperature,
            top_p=settings.llm_top_p,
            max_completion_tokens=settings.llm_max_tokens,
            frequency_penalty=settings.llm_frequency_penalty,
            timeout=settings.llm_timeout_seconds,
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        top_p=settings.llm_top_p,
        num_predict=settings.llm_max_tokens,  # Ollama's name for max_tokens
        client_kwargs={"timeout": settings.llm_timeout_seconds},
        format="json" if json_mode else None,
    )
