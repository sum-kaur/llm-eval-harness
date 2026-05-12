from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> tuple[str, int]:
        """Return (response_text, tokens_used)."""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        import anthropic
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> tuple[str, int]:
        msgs = messages or [{"role": "user", "content": prompt}]
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, messages=msgs)
        if system:
            kwargs["system"] = system
        if temperature != 0.0:
            kwargs["temperature"] = temperature

        resp = await asyncio.wait_for(
            self._client.messages.create(**kwargs),
            timeout=timeout,
        )
        text = resp.content[0].text
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        return text, tokens


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        import openai
        self.model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> tuple[str, int]:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if messages:
            msgs.extend(messages)
        else:
            msgs.append({"role": "user", "content": prompt})

        resp = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens


def make_provider(provider_name: str, model: str, **kwargs) -> LLMProvider:
    if provider_name == "anthropic":
        return AnthropicProvider(model=model, api_key=kwargs.get("api_key"))
    if provider_name in ("openai", "openai_compatible"):
        return OpenAIProvider(
            model=model,
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
        )
    raise ValueError(f"Unknown provider: {provider_name!r}. Supported: anthropic, openai")
