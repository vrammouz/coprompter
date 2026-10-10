"""Render arbitrary structured records as chat prompts using format templates."""

from __future__ import annotations

from collections.abc import Mapping
import json
from string import Formatter


class _JSONValueFormatter(Formatter):
    def format_field(self, value: object, format_spec: str) -> str:
        if format_spec:
            return super().format_field(value, format_spec)
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Template value is not JSON-serializable: {error}") from error


class PromptSerializer:
    """Turn mapping records into chat messages using format-string templates.

    Templates use Python's format-string syntax. Top-level fields use
    ``{field}``; nested mappings and sequences use item access such as
    ``{customer[name]}`` or ``{documents[0]}``. Non-string values are rendered
    as JSON so structured source fields remain unambiguous in the prompt.
    """

    def __init__(
        self,
        user_template: str,
        system_template: str | None = None,
        assistant_template: str | None = None,
    ) -> None:
        if not user_template.strip():
            raise ValueError("user_template must not be empty")
        if assistant_template is not None and not assistant_template.strip():
            raise ValueError("assistant_template must not be empty when provided")
        self.user_template = user_template
        self.system_template = system_template
        self.assistant_template = assistant_template
        self._formatter = _JSONValueFormatter()

    def _render(self, template: str, record: Mapping[str, object]) -> str:
        try:
            return self._formatter.vformat(template, (), dict(record))
        except (AttributeError, IndexError, KeyError) as error:
            raise ValueError(f"Unable to render prompt template: {error}") from error

    def serialize(self, record: Mapping[str, object]) -> list[dict[str, str]]:
        """Return OpenAI-compatible chat messages rendered from one record."""
        messages: list[dict[str, str]] = []
        if self.system_template is not None:
            messages.append(
                {"role": "system", "content": self._render(self.system_template, record)}
            )
        messages.append(
            {"role": "user", "content": self._render(self.user_template, record)}
        )
        if self.assistant_template is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": self._render(self.assistant_template, record),
                }
            )
        return messages
