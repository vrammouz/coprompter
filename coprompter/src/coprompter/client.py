"""Reusable asynchronous chat client for an OpenAI-compatible vLLM server."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TypeAlias

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

ChatPrompt: TypeAlias = Sequence[ChatCompletionMessageParam]


class VLLMChatClient:
    """Submit chat prompts to a vLLM OpenAI-compatible API endpoint.

    Use the client as an async context manager to close its HTTP resources
    deterministically.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_concurrency: int = 8,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def complete(
        self,
        messages: ChatPrompt,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        """Generate one response and return its text content."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")

        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )

        if not response.choices:
            raise ValueError("The chat completion response contains no choices")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("The chat completion response contains no text content")
        return content

    async def complete_many(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        num_chunks: int = 30,
    ) -> list[str]:
        """Generate ordered responses, splitting only when prompts exceed num_chunks."""
        if num_chunks < 1:
            raise ValueError("num_chunks must be at least 1")

        if len(prompts) <= num_chunks:
            chunks = [prompts]
        else:
            quotient, remainder = divmod(len(prompts), num_chunks)
            chunks = []
            start = 0
            for chunk_index in range(num_chunks):
                chunk_length = quotient + (chunk_index < remainder)
                chunks.append(prompts[start : start + chunk_length])
                start += chunk_length

        responses: list[str] = []
        for chunk in chunks:
            chunk_responses = await asyncio.gather(
                *(
                    self.complete(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    )
                    for prompt in chunk
                )
            )
            responses.extend(chunk_responses)
        return responses

    async def close(self) -> None:
        """Close the underlying asynchronous HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> VLLMChatClient:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.close()
