"""LLM factory — returns ChatOllama or ChatOpenAI depending on INFERENCE_BACKEND."""

from __future__ import annotations

from typing import TYPE_CHECKING

from api.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# Explicit cap shared by both backends — vllm-metal can truncate long answers
# while Ollama defaults to unlimited (num_predict=-1).
MAX_TOKENS = 1024


def make_llm(temperature: float = 0.0, json_mode: bool = False) -> BaseChatModel:
    """Return the configured LLM backend.

    ollama → ChatOllama; vllm → ChatOpenAI pointing at the vllm/vllm-metal endpoint
    (api_key="EMPTY" is required by langchain but ignored by vLLM).

    temperature=0.0 by default for deterministic, reproducible eval.
    frequency_penalty=0.3 (vllm) guards against Qwen 2.5 degenerate loops at temperature=0;
    Ollama applies repeat_penalty=1.1 by default for the same reason.
    """
    if settings.inference_backend == "vllm":
        from langchain_openai import ChatOpenAI

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

    # Default: Ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        top_p=1.0,
        num_predict=MAX_TOKENS,  # Ollama's name for max_tokens
        format="json" if json_mode else "",
    )
