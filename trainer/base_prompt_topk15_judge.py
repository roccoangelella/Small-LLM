"""Judge only the ADR-0153 Base Prompt sampled_topk15 view through GemRouter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from trainer.base_prompt_judge import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    JUDGE_PROMPT_ID,
    JUDGE_SYSTEM_PROMPT,
    JUDGMENT_SCHEMA,
    MAX_BATCH_SIZE,
    GemRouterJudgeClient,
    _judge_rows,
    _objective_rows,
    _sha256_path,
    _summary,
    _targets,
)

VIEW = "sampled_topk15"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def judge_topk15_bundle(
    bundle: Mapping[str, object],
    *,
    client,
    source_sha256: str | None,
    health: Mapping[str, object] | None,
    batch_size: int,
    max_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, object]:
    target_results: dict[str, object] = {}
    for label, suite in _targets(bundle).items():
        selected = suite.get(VIEW)
        if not isinstance(selected, Mapping):
            raise ValueError(f"Base Prompt suite for {label!r} has no {VIEW!r} view")
        sampling = selected.get("sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError(f"Base Prompt {VIEW!r} view has no sampling contract")
        expected_sampling = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 15,
            "seed": 17,
        }
        if any(sampling.get(key) != value for key, value in expected_sampling.items()):
            raise ValueError(
                f"Base Prompt {VIEW!r} sampling contract does not match ADR 0153"
            )
        rows = _objective_rows(suite, VIEW)
        judgments, batches = _judge_rows(
            rows,
            client=client,
            batch_size=batch_size,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        target_results[label] = {
            "prompt_set_id": suite.get("suite_identity", {}).get("prompt_set_id")
            if isinstance(suite.get("suite_identity"), Mapping)
            else None,
            VIEW: {
                "summary": _summary(judgments),
                "cases": judgments,
            },
            "provider_batches": {VIEW: batches},
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
            "selected_views": [VIEW],
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
    result = judge_topk15_bundle(
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
    print(f"Saved GemRouter sampled_topk15 judgments to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VIEW", "judge_topk15_bundle", "main"]
