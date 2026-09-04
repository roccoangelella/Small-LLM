"""Post-process Base Prompt v2 generations with the project's GemRouter endpoint.

The GPU evaluator intentionally emits durable raw generation evidence only.
This module reads that JSON later and asks the same private GemRouter endpoint
used by the R-SFT data pipeline to judge the 100 objective prompts semantically.
The 20 qualitative continuations remain readable, unscored evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

JUDGMENT_SCHEMA = "small-llm-base-prompt-gemrouter-judgment-v1"
JUDGE_PROMPT_ID = "base-prompt-semantic-judge-v1"
DEFAULT_JUDGE_MODEL = "gemini-3.7-flash"
DEFAULT_BATCH_SIZE = 20
MAX_BATCH_SIZE = 25
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 120.0

JUDGE_SYSTEM_PROMPT = """You are an impartial semantic evaluator for a small language-model benchmark.

For each item, judge whether the model continuation correctly answers the ORIGINAL prompt, using reference_answer as the benchmark answer key.

Rules:
1. Judge semantic correctness, not literal substring overlap. Accept harmless wording variants, capitalization differences, units, or explanatory text when the intended answer is clearly equivalent to reference_answer.
2. The answer attributable to the ORIGINAL prompt must be correct. If the continuation later invents another question/answer, do not give credit merely because reference_answer appears in that unrelated later text.
3. A contradiction overrides incidental overlap. If the continuation gives an answer incompatible with reference_answer, mark it incorrect even if some characters or words from reference_answer occur elsewhere. For example, reference_answer "0" does not make "-40 C" correct.
4. If the continuation gives multiple incompatible answers to the original prompt, mark it incorrect.
5. If there is no clear answer to the original prompt, mark it incorrect.
6. For arithmetic, extraction, classification, and transformation tasks, require the requested result to be semantically correct; extra text is allowed only if it does not change or contradict that result.
7. Treat reference_answer as the benchmark key. Do not rewrite the benchmark or substitute a different fact.
8. Return strict RFC 8259 JSON only: one array, same item order, with exactly three fields per object: id, verdict, reason. verdict must be "correct" or "incorrect". reason must be a concise explanation of at most 20 words. No Markdown and no text outside the JSON array."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rsft_transport() -> ModuleType:
    path = _repo_root() / "post_training" / "R-SFT" / "dataset.py"
    module_name = "small_llm_shared_gemrouter_transport"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GemRouter transport from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _health_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, (path or "") + "/health", "", ""))


class GemRouterJudgeClient:
    """OpenAI-compatible GemRouter client with deterministic judge sampling."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("judge model must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        transport = _load_rsft_transport()
        self.api_key = transport.resolve_api_key()
        self.endpoint = transport.resolve_endpoint()
        self.model = model.strip()
        self.timeout_seconds = float(timeout_seconds)

    def health_gate(self) -> dict[str, object]:
        request = Request(
            _health_url(self.endpoint),
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=30.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - fail closed before judge traffic
            raise RuntimeError("GemRouter health gate failed before judge traffic") from error
        if not isinstance(payload, Mapping):
            raise RuntimeError("GemRouter health response is not an object")
        config: Mapping[str, Any] = payload
        if isinstance(payload.get("config"), Mapping):
            config = payload["config"]
        backend_order = config.get("backendOrder")
        fallback = config.get("fallbackEnabled")
        if backend_order != ["gemini-api"] or fallback is not False:
            raise RuntimeError(
                "GemRouter judge requires backendOrder=['gemini-api'] and fallbackEnabled=false"
            )
        return {
            "backendOrder": list(backend_order),
            "fallbackEnabled": bool(fallback),
        }

    def complete(self, messages: Sequence[Mapping[str, str]]) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0.0,
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
                f"GemRouter judge request failed with HTTP status {error.code}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                "GemRouter judge request failed before receiving a response"
            ) from error
        try:
            response_payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("GemRouter judge returned a non-JSON response") from error
        if not isinstance(response_payload, Mapping):
            raise RuntimeError("GemRouter judge response must be a JSON object")
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("GemRouter judge response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise RuntimeError("GemRouter judge first choice is not an object")
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise RuntimeError("GemRouter judge first choice has no text content")
        return {
            "content": str(message["content"]),
            "model": str(response_payload.get("model") or self.model),
            "finish_reason": (
                str(choice.get("finish_reason"))
                if choice.get("finish_reason") is not None
                else None
            ),
            "usage": (
                dict(response_payload["usage"])
                if isinstance(response_payload.get("usage"), Mapping)
                else None
            ),
        }


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Base Prompt judge row field {key!r} must be non-empty text")
    return value


def _objective_rows(prompt_suite: Mapping[str, object], view: str) -> list[dict[str, str]]:
    selected = prompt_suite.get(view)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Base Prompt suite has no {view!r} view")
    cases = selected.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Base Prompt {view!r} cases must be a list")
    rows: list[dict[str, str]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Base Prompt case must be an object")
        if bool(case.get("qualitative")):
            continue
        if case.get("scored") is not True:
            raise ValueError("objective Base Prompt case must have scored=true")
        judge_status = case.get("judge_status")
        if judge_status != "pending":
            raise ValueError(
                f"objective Base Prompt case must be pending judgment, got {judge_status!r}"
            )
        rows.append(
            {
                "id": _text(case, "name"),
                "family": _text(case, "family"),
                "prompt": _text(case, "prompt"),
                "reference_answer": _text(case, "reference_answer"),
                "continuation": (
                    str(case.get("continuation"))
                    if isinstance(case.get("continuation"), str)
                    else ""
                ),
            }
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Base Prompt {view!r} objective IDs are not unique")
    return rows


def build_messages(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    if not rows or len(rows) > MAX_BATCH_SIZE:
        raise ValueError(f"judge batch size must be in [1, {MAX_BATCH_SIZE}]")
    payload = [
        {
            "id": row["id"],
            "family": row["family"],
            "prompt": row["prompt"],
            "reference_answer": row["reference_answer"],
            "continuation": row["continuation"],
        }
        for row in rows
    ]
    return (
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    )


def parse_judgments(text: str, *, expected_ids: Sequence[str]) -> list[dict[str, object]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("judge response must be non-empty text")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("judge response must be strict JSON") from error
    if not isinstance(payload, list) or len(payload) != len(expected_ids):
        raise ValueError("judge response count does not match request")
    results: list[dict[str, object]] = []
    for index, (item, expected_id) in enumerate(zip(payload, expected_ids, strict=True)):
        if not isinstance(item, Mapping) or set(item) != {"id", "verdict", "reason"}:
            raise ValueError(
                f"judge item {index} must contain exactly id, verdict, reason"
            )
        item_id = item.get("id")
        verdict = item.get("verdict")
        reason = item.get("reason")
        if item_id != expected_id:
            raise ValueError(
                f"judge item {index} id {item_id!r} != expected {expected_id!r}"
            )
        if verdict not in {"correct", "incorrect"}:
            raise ValueError(f"judge item {index} has invalid verdict {verdict!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"judge item {index} reason must be non-empty text")
        results.append(
            {
                "name": expected_id,
                "verdict": verdict,
                "score": 1 if verdict == "correct" else 0,
                "reason": reason.strip(),
            }
        )
    return results


def _judge_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    client: Any,
    batch_size: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    judgments: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    total_batches = (len(rows) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = list(rows[start : start + batch_size])
        ids = [row["id"] for row in batch]
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.complete(build_messages(batch))
                content = response["content"] if isinstance(response, Mapping) else response.content
                parsed = parse_judgments(str(content), expected_ids=ids)
                for row, judgment in zip(batch, parsed, strict=True):
                    judgment["family"] = row["family"]
                judgments.extend(parsed)
                batches.append(
                    {
                        "batch": batch_index,
                        "attempt": attempt,
                        "cases": len(batch),
                        "provider_model": (
                            response.get("model")
                            if isinstance(response, Mapping)
                            else getattr(response, "model", None)
                        ),
                        "finish_reason": (
                            response.get("finish_reason")
                            if isinstance(response, Mapping)
                            else getattr(response, "finish_reason", None)
                        ),
                    }
                )
                print(
                    f"[base-prompt judge] batch {batch_index}/{total_batches} "
                    f"accepted ({len(judgments)}/{len(rows)} cases)",
                    flush=True,
                )
                break
            except Exception as error:  # noqa: BLE001 - retry provider/JSON failures
                last_error = error
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Base Prompt judge batch {batch_index} failed after {max_attempts} attempts"
                    ) from error
                time.sleep(retry_delay_seconds)
        if last_error is not None and len(judgments) < min(start + len(batch), len(rows)):
            raise RuntimeError("Base Prompt judge retry loop ended without a result") from last_error
    return judgments, batches


def _summary(judgments: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(judgments)
    correct = sum(int(row["score"]) for row in judgments)
    families = sorted({str(row["family"]) for row in judgments})
    by_family: dict[str, object] = {}
    for family in families:
        selected = [row for row in judgments if row["family"] == family]
        family_correct = sum(int(row["score"]) for row in selected)
        by_family[family] = {
            "cases": len(selected),
            "correct": family_correct,
            "accuracy": family_correct / len(selected) if selected else None,
        }
    return {
        "judged_cases": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
        "by_family": by_family,
    }


def judge_prompt_suite(
    prompt_suite: Mapping[str, object],
    *,
    client: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, object]:
    identity = prompt_suite.get("suite_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Base Prompt suite has no suite_identity")
    prompt_set_id = identity.get("prompt_set_id")
    if prompt_set_id != "base-prompt-v2-unique-120-2026-09-04":
        raise ValueError(
            "GemRouter judge only accepts the corrected unique-120 Base Prompt v2 set"
        )
    contract = prompt_suite.get("scoring_contract")
    if not isinstance(contract, Mapping) or contract.get("local_string_or_regex_scoring") is not False:
        raise ValueError("Base Prompt suite is not a raw semantic-judge artifact")

    result: dict[str, object] = {"prompt_set_id": prompt_set_id}
    all_batches: dict[str, object] = {}
    for view in ("greedy", "sampled"):
        rows = _objective_rows(prompt_suite, view)
        judgments, batches = _judge_rows(
            rows,
            client=client,
            batch_size=batch_size,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        result[view] = {
            "summary": _summary(judgments),
            "cases": judgments,
        }
        all_batches[view] = batches
    result["provider_batches"] = all_batches
    return result


def _targets(bundle: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    schema = bundle.get("schema")
    if schema == "small-llm-pretraining-base-prompts-v2":
        return {"base_prompt_suite_v2": bundle}
    if schema == "small-llm-pretraining-evaluation-v2":
        suite = bundle.get("base_prompt_suite_v2")
        if not isinstance(suite, Mapping):
            raise ValueError("pretraining evaluation bundle has no Base Prompt v2 section")
        return {"pretraining": suite}
    if schema == "small-llm-post-sft-qualification-v2":
        result: dict[str, Mapping[str, object]] = {}
        for label in ("parent", "sft"):
            side = bundle.get(label)
            if not isinstance(side, Mapping):
                raise ValueError(f"SFT evaluation bundle has no {label!r} section")
            scorecard = side.get("scorecard")
            if not isinstance(scorecard, Mapping):
                raise ValueError(f"SFT evaluation {label!r} has no scorecard")
            suite = scorecard.get("base_prompt_suite_v2")
            if not isinstance(suite, Mapping):
                raise ValueError(f"SFT evaluation {label!r} has no Base Prompt v2 section")
            result[label] = suite
        return result
    raise ValueError(f"unsupported evaluation schema for Base Prompt judging: {schema!r}")


def judge_bundle(
    bundle: Mapping[str, object],
    *,
    client: Any,
    source_sha256: str | None = None,
    health: Mapping[str, object] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, object]:
    target_results = {
        label: judge_prompt_suite(
            suite,
            client=client,
            batch_size=batch_size,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        for label, suite in _targets(bundle).items()
    }
    return {
        "schema": JUDGMENT_SCHEMA,
        "judge": {
            "transport": "GemRouter",
            "requested_model": getattr(client, "model", DEFAULT_JUDGE_MODEL),
            "temperature": 0.0,
            "prompt_id": JUDGE_PROMPT_ID,
            "prompt_sha256": hashlib.sha256(
                JUDGE_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "batch_size": batch_size,
            "max_attempts": max_attempts,
            "health_gate": dict(health) if isinstance(health, Mapping) else None,
        },
        "source": {
            "schema": bundle.get("schema"),
            "sha256": source_sha256,
        },
        "targets": target_results,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge Base Prompt v2 generations with the project's GemRouter endpoint."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        parser.error(f"--batch-size must be in [1, {MAX_BATCH_SIZE}]")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds cannot be negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    source_path = args.input.resolve()
    try:
        bundle = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read evaluation JSON {source_path}") from error
    if not isinstance(bundle, Mapping):
        raise RuntimeError("evaluation JSON root must be an object")

    client = GemRouterJudgeClient(
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    health = client.health_gate()
    result = judge_bundle(
        bundle,
        client=client,
        source_sha256=_sha256_path(source_path),
        health=health,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved GemRouter Base Prompt judgments to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "GemRouterJudgeClient",
    "JUDGE_PROMPT_ID",
    "JUDGE_SYSTEM_PROMPT",
    "JUDGMENT_SCHEMA",
    "build_messages",
    "judge_bundle",
    "judge_prompt_suite",
    "parse_judgments",
]
