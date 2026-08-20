"""Deterministic prompt-wrapper robustness probe for the production R-SFT model.

This is an additive diagnostic for the canonical R-SFT qualification.  It asks a
small representative subset of the frozen novel reasoning cases through three
wrappers: the trained chat template, raw Question/Answer text, and a plain
natural-language prompt.  Final-answer correctness and reasoning-token protocol
transfer are reported separately.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from statistics import mean
import tempfile
from typing import Any, Mapping, Sequence

import torch

import chat as chat_runtime
import post_training.rsft_eval_suite as rsft_suite
from post_training.sft.schema import ChatMessage
from post_training.sft.template import GPT2ChatTemplate
from trainer.identity import canonical_hash
from trainer.post_pretraining_prompt_suite import sample_token_ids

WRAPPERS = ("chat", "question_answer", "plain")
FULL_CASES_PER_SKILL = 2
FAST_CASES_PER_SKILL = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _selected_cases(suite: str):
    per_skill = FAST_CASES_PER_SKILL if suite == "fast" else FULL_CASES_PER_SKILL
    grouped: dict[str, list[Any]] = {}
    for case in rsft_suite.REASONING_CASES:
        grouped.setdefault(case.skill, []).append(case)
    return tuple(
        case
        for skill in ("INF", "DED", "REL", "CSP", "IND", "ABD", "MAG")
        for case in grouped[skill][:per_skill]
    )


def _wrapper_prompt_ids(
    wrapper: str,
    *,
    case: Any,
    encoder: Any,
    max_seq_len: int,
) -> tuple[int, ...]:
    if wrapper == "chat":
        template = GPT2ChatTemplate(
            maximum_context_tokens=max_seq_len,
            maximum_assistant_tokens=max_seq_len,
        )
        return template.encode_generation_prompt(
            (ChatMessage("user", case.prompt),),
            encoder,
        )
    if wrapper == "question_answer":
        text = f"Question: {case.prompt}\nAnswer:"
    elif wrapper == "plain":
        text = f"{case.prompt}\n"
    else:
        raise ValueError(f"unknown wrapper: {wrapper}")
    token_ids = tuple(int(token) for token in encoder.encode(text))
    if len(token_ids) >= max_seq_len:
        raise ValueError(f"{wrapper} prompt exceeds model context")
    return token_ids


def _generation_row(
    model: torch.nn.Module,
    *,
    wrapper: str,
    case: Any,
    case_index: int,
    encoder: Any,
    max_seq_len: int,
    precision: str,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt_ids = _wrapper_prompt_ids(
        wrapper,
        case=case,
        encoder=encoder,
        max_seq_len=max_seq_len,
    )
    budget = min(max_new_tokens, max_seq_len - len(prompt_ids))
    generated = sample_token_ids(
        model,
        prompt_ids,
        max_new_tokens=budget,
        max_seq_len=max_seq_len,
        eos_token_id=rsft_suite.EOS_TOKEN_ID,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=17 + case_index * 1000,
        precision=precision,
    )
    protocol = rsft_suite._protocol_fields(generated, encoder)
    terminated = bool(generated and generated[-1] == rsft_suite.EOS_TOKEN_ID)
    body = generated[:-1] if terminated else generated
    continuation = encoder.decode(body).strip()
    protocol_answer = str(protocol["answer_text"])
    answer = protocol_answer if protocol_answer else continuation
    answer_correct = rsft_suite._reasoning_answer_correct(case, answer)
    strict_correct = bool(protocol["well_formed"]) and rsft_suite._reasoning_answer_correct(
        case,
        protocol_answer,
    )
    return {
        "name": case.name,
        "skill": case.skill,
        "level": case.level,
        "prompt": case.prompt,
        "wrapper": wrapper,
        "seed": 17 + case_index * 1000,
        "prompt_tokens": len(prompt_ids),
        "continuation": continuation,
        "answer": answer,
        "answer_correct_any_format": answer_correct,
        "strict_correct": strict_correct,
        "protocol": {
            key: value
            for key, value in protocol.items()
            if not key.endswith("_token_ids")
        },
        "generated_token_ids": generated,
    }


def _wrapper_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    protocol_rows = [
        row["protocol"]
        for row in rows
        if isinstance(row.get("protocol"), Mapping)
    ]
    return {
        "cases": len(rows),
        "answer_accuracy_any_format": mean(
            float(bool(row["answer_correct_any_format"])) for row in rows
        ),
        "strict_protocol_answer_accuracy": mean(
            float(bool(row["strict_correct"])) for row in rows
        ),
        "reasoning_start_rate": mean(
            float(
                isinstance(row.get("protocol"), Mapping)
                and int(row["protocol"].get("reasoning_start_count", 0)) >= 1
            )
            for row in rows
        ),
        "protocol": rsft_suite._protocol_summary(protocol_rows),
        "runs": list(rows),
    }


def _delta(value: object, baseline: object) -> float | None:
    if isinstance(value, bool) or isinstance(baseline, bool):
        return None
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return float(value) - float(baseline)


def evaluate_prompt_wrapper_robustness(
    model: torch.nn.Module,
    *,
    encoder: Any,
    max_seq_len: int,
    precision: str,
    suite: str,
    max_new_tokens: int,
) -> dict[str, object]:
    cases = _selected_cases(suite)
    results: dict[str, dict[str, object]] = {}
    for wrapper in WRAPPERS:
        rows: list[dict[str, object]] = []
        for case_index, case in enumerate(cases):
            print(
                f"[rsft-wrapper] {wrapper} {case_index + 1}/{len(cases)} {case.name}",
                flush=True,
            )
            rows.append(
                _generation_row(
                    model,
                    wrapper=wrapper,
                    case=case,
                    case_index=case_index,
                    encoder=encoder,
                    max_seq_len=max_seq_len,
                    precision=precision,
                    max_new_tokens=max_new_tokens,
                )
            )
        results[wrapper] = _wrapper_summary(rows)

    chat = results["chat"]
    deltas: dict[str, dict[str, float | None]] = {}
    for wrapper in ("question_answer", "plain"):
        current = results[wrapper]
        current_protocol = current.get("protocol")
        chat_protocol = chat.get("protocol")
        deltas[wrapper] = {
            "answer_accuracy_any_format": _delta(
                current.get("answer_accuracy_any_format"),
                chat.get("answer_accuracy_any_format"),
            ),
            "strict_protocol_answer_accuracy": _delta(
                current.get("strict_protocol_answer_accuracy"),
                chat.get("strict_protocol_answer_accuracy"),
            ),
            "reasoning_start_rate": _delta(
                current.get("reasoning_start_rate"),
                chat.get("reasoning_start_rate"),
            ),
            "well_formed_rate": _delta(
                current_protocol.get("well_formed_rate")
                if isinstance(current_protocol, Mapping)
                else None,
                chat_protocol.get("well_formed_rate")
                if isinstance(chat_protocol, Mapping)
                else None,
            ),
        }

    return {
        "schema": "small-llm-rsft-prompt-wrapper-robustness-v1",
        "case_count": len(cases),
        "cases_per_skill": FAST_CASES_PER_SKILL if suite == "fast" else FULL_CASES_PER_SKILL,
        "case_identity": canonical_hash([asdict(case) for case in cases]),
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "samples_per_problem": 1,
            "max_new_tokens": max_new_tokens,
            "seed": 17,
        },
        "wrapper_contracts": {
            "chat": "small-llm-s0-v1 chat generation prompt ending at Assistant:\\n",
            "question_answer": "Question: {problem}\\nAnswer:",
            "plain": "{problem}\\n",
        },
        "wrappers": results,
        "delta_vs_chat": deltas,
        "interpretation": (
            "Use protocol transfer and answer correctness as separate axes. A drop under raw wrappers "
            "measures prompt-template dependence; it does not by itself invalidate chat-conditioned reasoning."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=256)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--rsft-checkpoint-dir", type=Path)
    parser.add_argument("--rsft-repo-id")
    parser.add_argument("--rsft-run-id", default="100m-2b-rsft-r0-12306-001")
    parser.add_argument("--rsft-pointer", choices=("best", "latest"), default="latest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != "small-llm-post-rsft-qualification":
        raise RuntimeError("prompt-wrapper probe requires a canonical R-SFT qualification report")
    if not args.rsft_checkpoint_dir and not args.rsft_repo_id:
        raise RuntimeError("pass --rsft-checkpoint-dir or --rsft-repo-id")

    device = rsft_suite._resolve_device(args.device)
    precision = rsft_suite._resolve_precision(args.precision, device)
    token = os.environ.get(args.token_env)
    with tempfile.TemporaryDirectory(prefix="small-llm-rsft-wrapper-") as temporary_dir:
        resolve_args = argparse.Namespace(
            rsft_checkpoint_dir=args.rsft_checkpoint_dir,
            rsft_repo_id=args.rsft_repo_id,
            rsft_run_id=args.rsft_run_id,
            rsft_pointer=args.rsft_pointer,
        )
        model, config, _identity, _transport, reasoning_spec = rsft_suite._resolve_rsft(
            resolve_args,
            token=token,
            temporary=Path(temporary_dir),
            device=device,
        )
        normal_encoding = rsft_suite._normal_encoding()
        tokenizer_module = chat_runtime._load_rsft_tokenizer_module()
        encoder = tokenizer_module.ReasoningGPT2Encoder(
            reasoning_spec,
            base_encoding=normal_encoding,
        )
        result = evaluate_prompt_wrapper_robustness(
            model,
            encoder=encoder,
            max_seq_len=config.max_seq_len,
            precision=precision,
            suite=args.suite,
            max_new_tokens=args.max_new_tokens,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rsft = report.get("rsft")
    if not isinstance(rsft, dict) or not isinstance(rsft.get("scorecard"), dict):
        raise RuntimeError("canonical report is missing rsft.scorecard")
    rsft["scorecard"]["prompt_wrapper_robustness"] = result
    sampling = report.get("sampling_contracts")
    if isinstance(sampling, dict):
        sampling["prompt_wrapper_robustness"] = dict(result["sampling"])
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_hash(report)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(f"Appended prompt-wrapper robustness to: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
