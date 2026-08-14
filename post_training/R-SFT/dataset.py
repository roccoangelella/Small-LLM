"""Thin GemRouter/Gemini transport for the first reasoning-SFT data lane.

This module intentionally owns only API request/response plumbing. Prompt
construction, skill/difficulty sampling, deterministic verification, rejection,
deduplication, and final dataset serialization are deliberately deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GEMR_API_KEY_ENV = "GEMR_API_KEY"
DEFAULT_ENDPOINT = "https://gemr.84-8-255-231.nip.io/v1/chat/completions"
DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One OpenAI-compatible chat message sent to the distillation teacher."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("message role must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")

    def as_payload(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class DistillationResponse:
    """Normalized teacher output plus the untouched provider response."""

    content: str
    model: str | None
    finish_reason: str | None
    usage: Mapping[str, Any] | None
    raw: Mapping[str, Any]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dotenv_value(path: Path, key: str) -> str | None:
    """Read one simple KEY=value entry without adding a dotenv dependency."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value or None
    return None


def resolve_api_key(
    api_key: str | None = None,
    *,
    env_path: Path | None = None,
) -> str:
    """Resolve the GemRouter key from an explicit value, env, or repo .env."""

    if api_key is not None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        return api_key.strip()

    environment_value = os.environ.get(GEMR_API_KEY_ENV)
    if environment_value and environment_value.strip():
        return environment_value.strip()

    dotenv_path = _repository_root() / ".env" if env_path is None else Path(env_path)
    dotenv_value = _dotenv_value(dotenv_path, GEMR_API_KEY_ENV)
    if dotenv_value:
        return dotenv_value

    raise RuntimeError(
        f"{GEMR_API_KEY_ENV} is required; set it in the environment or repository .env"
    )


def _normalize_messages(
    messages: Sequence[ChatMessage | Mapping[str, str]],
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("at least one chat message is required")

    normalized: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            normalized.append(message.as_payload())
            continue
        if not isinstance(message, Mapping):
            raise TypeError("messages must contain ChatMessage or mapping values")
        try:
            role = message["role"]
            content = message["content"]
        except KeyError as error:
            raise ValueError("message mappings require role and content") from error
        normalized.append(ChatMessage(role=role, content=content).as_payload())
    return normalized


def _parse_response(payload: Any) -> DistillationResponse:
    if not isinstance(payload, Mapping):
        raise RuntimeError("GemRouter response must be a JSON object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("GemRouter response has no choices")

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("GemRouter first choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("GemRouter first choice has no message")

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("GemRouter assistant content must be a string")

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        model = str(model)

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)

    usage = payload.get("usage")
    normalized_usage = dict(usage) if isinstance(usage, Mapping) else None

    return DistillationResponse(
        content=content,
        model=model,
        finish_reason=finish_reason,
        usage=normalized_usage,
        raw=dict(payload),
    )


class GeminiDistillationClient:
    """Minimal OpenAI-compatible client for GemRouter-backed Gemini calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        env_path: Path | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.api_key = resolve_api_key(api_key, env_path=env_path)
        self.endpoint = endpoint.strip()
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)

    def complete(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
    ) -> DistillationResponse:
        """Send one chat completion using provider-default sampling parameters."""

        payload = {
            "model": self.model,
            "messages": _normalize_messages(messages),
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_bytes = response.read()
        except HTTPError as error:
            raise RuntimeError(
                f"GemRouter request failed with HTTP status {error.code}"
            ) from error
        except URLError as error:
            raise RuntimeError("GemRouter request failed before receiving a response") from error

        try:
            response_payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("GemRouter returned a non-JSON response") from error
        return _parse_response(response_payload)

    def complete_text(self, prompt: str) -> DistillationResponse:
        """Convenience wrapper for one user prompt."""

        return self.complete((ChatMessage(role="user", content=prompt),))


__all__ = [
    "ChatMessage",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "DistillationResponse",
    "GEMR_API_KEY_ENV",
    "GeminiDistillationClient",
    "resolve_api_key",
]
