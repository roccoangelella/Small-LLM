"""Small OpenAI-compatible client for the locally configured GemRouter."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dataset import config

from .storage import read_json


def _gemrouter_api_key() -> str:
    """Read an explicit environment override or Pi's existing local credential."""

    key = os.environ.get(config.GEMROUTER_API_KEY_ENV)
    if key:
        return key
    try:
        auth = read_json(config.GEMROUTER_PI_AUTH_PATH)
        key = auth.get("gemr", {}).get("key")
    except (FileNotFoundError, json.JSONDecodeError):
        key = None
    if not isinstance(key, str) or not key:
        raise RuntimeError(
            f"No GemRouter credential found. Set {config.GEMROUTER_API_KEY_ENV} or configure "
            f"Pi at {config.GEMROUTER_PI_AUTH_PATH}."
        )
    return key


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model's JSON-only response, tolerating a Markdown fence."""

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model response did not contain a JSON object")


def gemrouter_json(
    prompt: str,
    max_tokens: int,
    *,
    validator: Callable[[dict[str, Any]], None] | None = None,
    trace_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call Gemini 3.6 Flash through the local OpenAI-compatible GemRouter.

    ``trace_events`` is an opt-in, caller-owned record of every request attempt.
    It contains the model payload, raw model text, parsed JSON, or failure, but
    intentionally never the authorization header or API key. It is useful for
    bounded review tests and is left disabled for normal production stages.
    """

    body = {
        "model": config.GEMROUTER_MODEL,
        "temperature": config.LLM_REVIEW_TEMPERATURE,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "Return only a valid JSON object. Do not use Markdown."},
            {"role": "user", "content": prompt},
        ],
    }
    encoded = json.dumps(body).encode("utf-8")
    request = Request(
        config.GEMROUTER_BASE_URL.rstrip("/") + "/chat/completions",
        data=encoded,
        headers={
            "Content-Type": "application/json",
            config.GEMROUTER_AUTH_HEADER: f"Bearer {_gemrouter_api_key()}",
        },
        method="POST",
    )
    failures: list[str] = []
    for attempt in range(1, config.LLM_RETRY_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=config.GEMROUTER_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str):
                raise ValueError("GemRouter returned a non-text message")
            result = parse_json_object(content)
            if validator is not None:
                validator(result)
            if trace_events is not None:
                trace_events.append(
                    {
                        "attempt": attempt,
                        "request": body,
                        "raw_response": content,
                        "parsed_response": result,
                    }
                )
            return result
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError) as error:
            failure = f"attempt {attempt}: {type(error).__name__}: {error}"
            failures.append(failure)
            if trace_events is not None:
                trace_events.append({"attempt": attempt, "request": body, "error": failure})
            if attempt < config.LLM_RETRY_ATTEMPTS:
                time.sleep(config.LLM_RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError("GemRouter review failed: " + " | ".join(failures))
