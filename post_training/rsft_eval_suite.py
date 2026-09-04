"""Comprehensive S0-versus-R-SFT qualification for the 100M/2B model.

The suite keeps the frozen post-SFT regression matrix and adds a reasoning-aware
layer for the production atomic <think>...</think><answer>... protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
import json
import os
from pathlib import Path
import re
from statistics import mean
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

import chat as chat_runtime
from post_training.sft.behavior_eval import BEHAVIOR_CASES, evaluate_behavior, verify_response
from post_training.sft.bundle import verify_bundle
from post_training.sft.checkpoints import download_parent_checkpoint, load_verified_native_checkpoint
from post_training.sft.schema import ChatMessage
from post_training.sft.storage import SFTShardReader
from post_training.sft.template import GPT2ChatTemplate, TiktokenGPT2Encoder
from trainer.eval_entrypoint import ensure_eval_core
from trainer.eval_suite import evaluate_split, run_prompt_cases
from trainer.evaluation import evaluate_batches
from trainer.identity import canonical_hash
from trainer.post_pretraining_prompt_suite import PROMPT_CASES, sample_token_ids

BASE_SEMANTIC_VOCAB_SIZE = 50_257
RSFT_SEMANTIC_VOCAB_SIZE = 50_260
EOS_TOKEN_ID = 50_256
REASONING_START_TOKEN_ID = 50_257
REASONING_END_TOKEN_ID = 50_258
ANSWER_START_TOKEN_ID = 50_259
DEFAULT_REASONING_MAX_NEW_TOKENS = 256
DEFAULT_REASONING_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class ReasoningCase:
    name: str
    skill: str
    level: str
    prompt: str
    answer_regex: str


# Frozen, self-contained, mechanically scored probes. They intentionally avoid
# outside knowledge and mirror R0's seven reasoning skills without reproducing
# the examples committed in post_training/R-SFT/prompts.py.
REASONING_CASES: tuple[ReasoningCase, ...] = (
    # INF — immediate inference
    ReasoningCase("inf_glass", "INF", "L1", "Every glass badge in the tray is fragile. Badge K is a glass badge in the tray. What must be true about Badge K?", r"\bfragile\b"),
    ReasoningCase("inf_archive", "INF", "L1", "Only approved files may be archived. File R has been archived. What must be true about File R?", r"\bapproved\b"),
    ReasoningCase("inf_gate", "INF", "L2", "A gate opens if it has a green key or a silver key. Gate P has a green key and no silver key. Does Gate P open?", r"\b(?:yes|opens?)\b"),
    ReasoningCase("inf_coating", "INF", "L2", "No copper tags are waterproof. Tag M is copper. What follows about Tag M?", r"\b(?:not waterproof|isn['’]?t waterproof|cannot be waterproof)\b"),
    ReasoningCase("inf_qualify", "INF", "L3", "A sample qualifies exactly when it is sealed and sterile. Sample Q is sealed and sterile. Does Sample Q qualify?", r"\b(?:yes|qualif(?:y|ies|ied))\b"),
    # DED — multi-step deduction
    ReasoningCase("ded_sensor", "DED", "L1", "If a sensor is armed, its lamp is blue. If its lamp is blue, the control room receives a signal. Sensor T is armed. What follows?", r"\bcontrol room\b.*\b(?:receives?|gets?)\b.*\bsignal\b|\bsignal\b.*\bcontrol room\b"),
    ReasoningCase("ded_alarm", "DED", "L2", "If the alarm is active, the west door is locked. The west door is not locked. What follows about the alarm?", r"\b(?:alarm is not active|alarm isn['’]?t active|alarm is inactive)\b"),
    ReasoningCase("ded_route", "DED", "L2", "Exactly one route, North or South, is open. The North route is closed. Which route is open?", r"\bSouth\b"),
    ReasoningCase("ded_badge", "DED", "L3", "Every bronze badge grants lab access. Anything that grants lab access activates the inner scanner. Badge V is bronze. What must Badge V activate?", r"\binner scanner\b"),
    ReasoningCase("ded_ticket", "DED", "L3", "If a ticket is urgent, it is reviewed today. Every ticket reviewed today receives a timestamp. Ticket H is urgent. What must Ticket H receive?", r"\btimestamp\b"),
    # REL — relation composition
    ReasoningCase("rel_height", "REL", "L1", "Mira is taller than Niko. Niko is taller than Oren. Who is taller, Mira or Oren?", r"\bMira\b"),
    ReasoningCase("rel_time", "REL", "L1", "Task A happens before Task B. Task B happens before Task C. Which happens first, Task A or Task C?", r"\bTask A\b|\bA\b"),
    ReasoningCase("rel_containment", "REL", "L2", "The blue folder is inside Drawer J. Drawer J is inside Cabinet L. Where is the blue folder relative to Cabinet L?", r"\binside\b.*\bCabinet L\b|\bCabinet L\b.*\bcontains?\b"),
    ReasoningCase("rel_left", "REL", "L2", "Marker P is left of Marker Q. Marker Q is left of Marker R. Where is Marker P relative to Marker R?", r"\bleft\b"),
    ReasoningCase("rel_priority", "REL", "L3", "Job X has higher priority than Job Y, and Job Z has lower priority than Job Y. Which job has the highest priority?", r"\bJob X\b|\bX\b"),
    # CSP — compact constraint satisfaction
    ReasoningCase("csp_seats", "CSP", "L1", "Ari, Bea, and Cy occupy seats 1, 2, and 3 with no shared seats. Bea is in seat 1. Ari is not in seat 3. Which seat must Ari occupy?", r"\bseat 2\b|\b2\b"),
    ReasoningCase("csp_colors", "CSP", "L2", "Three boxes P, Q, and R are colored red, blue, and green, one color each. P is not red. Q is blue. R is not green. What color is P?", r"\bgreen\b"),
    ReasoningCase("csp_days", "CSP", "L2", "Ivo, June, and Kai present on Monday, Tuesday, and Wednesday, one person per day. June presents Tuesday. Ivo presents before Kai. Which day does Ivo present?", r"\bMonday\b"),
    ReasoningCase("csp_tools", "CSP", "L3", "Lena, Omar, and Pia each choose one tool: drill, saw, or wrench. Omar chooses the saw. Lena does not choose the drill. Pia does not choose the wrench. Which tool does Lena choose?", r"\bwrench\b"),
    ReasoningCase("csp_order", "CSP", "L3", "Cards A, B, and C are placed first, second, and third. A is before B. C is not third. B is not second. What is the order?", r"\bC\b.*\bA\b.*\bB\b"),
    # IND — controlled induction
    ReasoningCase("ind_shape", "IND", "L1", "A sorter uses exactly one property, shape or color. Red circles are L, blue circles are L, red triangles are M, and blue triangles are M. What label should a green triangle receive?", r"(?:^|\blabel\s+)M\b"),
    ReasoningCase("ind_size", "IND", "L2", "A device assigns A or B using exactly one property: size or material. Small wood=A, small metal=A, large wood=B, large metal=B. What is a large plastic item assigned?", r"(?:^|\bassigned\s+)B\b"),
    ReasoningCase("ind_border", "IND", "L2", "A classifier uses exactly one property: border or fill. Solid-border white cards are X, solid-border black cards are X, dashed-border white cards are Y, dashed-border black cards are Y. What label gets a dashed-border gray card?", r"(?:^|\blabel\s+)Y\b"),
    ReasoningCase("ind_switch", "IND", "L3", "A controller outputs ON or OFF based on exactly one property: symbol or background. Star/red=ON, star/blue=ON, square/red=OFF, square/blue=OFF. What output should star/green produce?", r"\bON\b"),
    ReasoningCase("ind_texture", "IND", "L3", "A bin uses exactly one property: texture or color. Smooth red goes to bin 1, smooth blue to bin 1, rough red to bin 2, rough blue to bin 2. Which bin gets a rough yellow object?", r"\bbin 2\b|^\s*2\b"),
    # ABD — closed-set abduction
    ReasoningCase("abd_fault", "ABD", "L1", "Exactly one fault occurred. Fault A causes only a beep. Fault B causes only a red light. Fault C causes both. The device beeps and shows a red light. Which fault occurred?", r"\bFault C\b|^\s*C\b"),
    ReasoningCase("abd_ink", "ABD", "L2", "Exactly one cartridge leaked. Cartridge P leaves a blue spot, Q leaves a yellow spot, and R leaves both blue and yellow spots. Both colors are observed. Which cartridge leaked?", r"\bCartridge R\b|^\s*R\b"),
    ReasoningCase("abd_network", "ABD", "L2", "One network issue occurred. Issue A causes slow upload only. Issue B causes slow download only. Issue C causes both slow upload and slow download. Both are slow. Which issue fits the evidence?", r"\bIssue C\b|^\s*C\b"),
    ReasoningCase("abd_lamps", "ABD", "L3", "One switch is faulty. Switch J makes lamp 1 fail only; K makes lamp 2 fail only; L makes lamps 1 and 2 fail. Both lamps fail. Which switch is faulty?", r"\bSwitch L\b|^\s*L\b"),
    ReasoningCase("abd_flags", "ABD", "L3", "Exactly one mode is active. Mode U raises flag alpha only. Mode V raises flag beta only. Mode W raises alpha and beta. Both flags are raised. Which mode is active?", r"\bMode W\b|^\s*W\b"),
    # MAG — numerical magnitude / bounds
    ReasoningCase("mag_chain", "MAG", "L1", "Quantity A is greater than Quantity B, and Quantity B is greater than 20. What must be true about Quantity A relative to 20?", r"\bgreater than 20\b|\babove 20\b"),
    ReasoningCase("mag_negative", "MAG", "L1", "Number X is positive and Number Y is negative. Which number is larger?", r"\b(?:Number )?X\b"),
    ReasoningCase("mag_bounds", "MAG", "L2", "Value P is at most 8. Value Q is at least 12. Which value must be larger?", r"\b(?:Value )?Q\b"),
    ReasoningCase("mag_order", "MAG", "L2", "R is less than S, and S is less than T. Which is the greatest of R, S, and T?", r"(?:^|\bgreatest\s+(?:is\s+)?)T\b"),
    ReasoningCase("mag_range", "MAG", "L3", "A score lies between 30 and 40. Another score is greater than 50. Which score must be larger?", r"\b(?:second|another) score\b|\bgreater than 50\b"),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_precision(value: str, device: torch.device) -> str:
    if value == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if value in {"fp16", "bf16"} and device.type != "cuda":
        raise RuntimeError(f"{value} evaluation requires CUDA")
    return value


def _config_values(config: object) -> dict[str, object]:
    if is_dataclass(config):
        return dict(asdict(config))
    raw = getattr(config, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    raise RuntimeError("model config is not introspectable")


def _assert_transition_compatible(s0_config: object, rsft_config: object) -> None:
    s0 = _config_values(s0_config)
    rsft = _config_values(rsft_config)
    if s0.get("semantic_vocab_size") != BASE_SEMANTIC_VOCAB_SIZE:
        raise RuntimeError("S0 checkpoint does not use the frozen GPT-2 semantic vocabulary")
    if rsft.get("semantic_vocab_size") != RSFT_SEMANTIC_VOCAB_SIZE:
        raise RuntimeError("R-SFT checkpoint does not use the frozen GPT-2 + 3-token vocabulary")
    for key in sorted(set(s0) | set(rsft)):
        if key == "semantic_vocab_size":
            continue
        if s0.get(key) != rsft.get(key):
            raise RuntimeError(
                f"S0/R-SFT model geometry differs at {key}: {s0.get(key)!r} != {rsft.get(key)!r}"
            )


def _masked_split(
    model: torch.nn.Module,
    *,
    precision: str,
    bundle_root: Path,
    split: str,
    maximum_blocks: int,
    microbatch_size: int,
) -> dict[str, object]:
    engine = SimpleNamespace(
        model=model,
        device=next(model.parameters()).device,
        config=SimpleNamespace(precision=precision),
        optimizer=None,
        best_validation_loss=None,
    )
    reader = SFTShardReader(bundle_root / split, split=split)
    return evaluate_batches(
        engine,
        reader.iter_from_start(),
        maximum_batches=maximum_blocks,
        microbatch_size=microbatch_size,
    )


def _normal_encoding():
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("R-SFT qualification requires tiktoken") from error
    return tiktoken.get_encoding("gpt2")


def _protocol_fields(ids: Sequence[int], rsft_encoder: Any) -> dict[str, object]:
    values = [int(token) for token in ids]
    terminated = bool(values and values[-1] == EOS_TOKEN_ID)
    body = values[:-1] if terminated else values
    counts = Counter(body)
    starts = [index for index, token in enumerate(body) if token == REASONING_START_TOKEN_ID]
    ends = [index for index, token in enumerate(body) if token == REASONING_END_TOKEN_ID]
    answers = [index for index, token in enumerate(body) if token == ANSWER_START_TOKEN_ID]
    ordered = bool(starts and ends and answers and starts[0] < ends[0] < answers[0])
    well_formed = (
        counts[REASONING_START_TOKEN_ID] == 1
        and counts[REASONING_END_TOKEN_ID] == 1
        and counts[ANSWER_START_TOKEN_ID] == 1
        and ordered
    )
    if ordered:
        reasoning_ids = body[starts[0] + 1 : ends[0]]
        answer_ids = body[answers[0] + 1 :]
    else:
        reasoning_ids = []
        answer_ids = []
    return {
        "well_formed": well_formed,
        "ordered": ordered,
        "reasoning_start_count": counts[REASONING_START_TOKEN_ID],
        "reasoning_end_count": counts[REASONING_END_TOKEN_ID],
        "answer_start_count": counts[ANSWER_START_TOKEN_ID],
        "reasoning_token_ids": reasoning_ids,
        "answer_token_ids": answer_ids,
        "reasoning_text": rsft_encoder.decode(reasoning_ids).strip() if reasoning_ids else "",
        "answer_text": rsft_encoder.decode(answer_ids).strip() if answer_ids else "",
        "reasoning_tokens": len(reasoning_ids),
        "answer_tokens": len(answer_ids),
        "terminated_with_eos": terminated,
        "runaway": not terminated,
        "nonempty_reasoning": bool(reasoning_ids),
        "nonempty_answer": bool(answer_ids),
    }


def _protocol_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"cases": 0}
    return {
        "cases": len(rows),
        "well_formed_rate": mean(float(bool(row["well_formed"])) for row in rows),
        "ordered_rate": mean(float(bool(row["ordered"])) for row in rows),
        "single_think_start_rate": mean(float(row["reasoning_start_count"] == 1) for row in rows),
        "single_think_end_rate": mean(float(row["reasoning_end_count"] == 1) for row in rows),
        "single_answer_start_rate": mean(float(row["answer_start_count"] == 1) for row in rows),
        "nonempty_reasoning_rate": mean(float(bool(row["nonempty_reasoning"])) for row in rows),
        "nonempty_answer_rate": mean(float(bool(row["nonempty_answer"])) for row in rows),
        "eos_termination_rate": mean(float(bool(row["terminated_with_eos"])) for row in rows),
        "runaway_rate": mean(float(bool(row["runaway"])) for row in rows),
        "mean_reasoning_tokens": mean(float(row["reasoning_tokens"]) for row in rows),
        "mean_answer_tokens": mean(float(row["answer_tokens"]) for row in rows),
    }


def _base_prompts_rsft(
    model: torch.nn.Module,
    *,
    rsft_encoder: Any,
    max_seq_len: int,
    precision: str,
    suite: str,
    sampled: bool,
) -> list[dict[str, object]]:
    encoding = _normal_encoding()
    cases = PROMPT_CASES[:8] if suite == "fast" else PROMPT_CASES
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        prompt_ids = encoding.encode(case.prompt, disallowed_special=())
        budget = case.max_new_tokens if sampled else min(case.max_new_tokens, 32)
        generated = sample_token_ids(
            model,
            prompt_ids,
            max_new_tokens=budget,
            max_seq_len=max_seq_len,
            eos_token_id=EOS_TOKEN_ID,
            temperature=1.0 if sampled else 0.0,
            top_p=1.0,
            top_k=0,
            seed=17 + index * 1000,
            precision=precision,
        )
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "sample": 1,
                "seed": 17 + index * 1000,
                "prompt": case.prompt,
                "prompt_token_ids": list(prompt_ids),
                "generated_token_ids": generated,
                "continuation": rsft_encoder.decode(
                    [token for token in generated if token != EOS_TOKEN_ID]
                ),
            }
        )
    return rows


def _evaluate_rsft_behavior(
    model: torch.nn.Module,
    *,
    rsft_encoder: Any,
    precision: str,
    max_seq_len: int,
    suite: str,
    reasoning_allowance: int,
) -> dict[str, object]:
    template = GPT2ChatTemplate(
        maximum_context_tokens=max_seq_len,
        maximum_assistant_tokens=max_seq_len,
    )
    cases = BEHAVIOR_CASES[:16] if suite == "fast" else BEHAVIOR_CASES
    rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        prompt_ids = template.encode_generation_prompt(case.messages, rsft_encoder)
        generated = sample_token_ids(
            model,
            prompt_ids,
            max_new_tokens=min(
                max_seq_len - len(prompt_ids),
                case.max_new_tokens + reasoning_allowance,
            ),
            max_seq_len=max_seq_len,
            eos_token_id=EOS_TOKEN_ID,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=17 + index,
            precision=precision,
        )
        protocol = _protocol_fields(generated, rsft_encoder)
        protocol_rows.append(protocol)
        answer_ids = protocol["answer_token_ids"]
        assert isinstance(answer_ids, list)
        verdict = verify_response(
            case,
            text=str(protocol["answer_text"]),
            response_token_ids=answer_ids,
            terminated_with_eos=bool(protocol["terminated_with_eos"]),
        )
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                **verdict,
                "protocol": {
                    key: value
                    for key, value in protocol.items()
                    if not key.endswith("_token_ids")
                },
                "generated_token_ids": generated,
            }
        )
    categories: dict[str, dict[str, object]] = {}
    for category in sorted({str(row["category"]) for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        categories[category] = {
            "cases": len(selected),
            "passed": sum(bool(row["passed"]) for row in selected),
            "pass_rate": mean(float(bool(row["passed"])) for row in selected),
        }
    return {
        "schema": "small-llm-rsft-behavior-v1",
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "passed": sum(bool(row["passed"]) for row in rows),
            "pass_rate": mean(float(bool(row["passed"])) for row in rows),
            "eos_termination_rate": mean(float(bool(row["terminated_with_eos"])) for row in rows),
            "runaway_rate": mean(float(bool(row["runaway"])) for row in rows),
            "empty_rate": mean(float(bool(row["empty"])) for row in rows),
            "role_leak_rate": mean(float(bool(row["role_leak"])) for row in rows),
            "mean_response_tokens": mean(float(row["response_tokens"]) for row in rows),
            "mean_trigram_repetition": mean(float(row["trigram_repetition"]) for row in rows),
        },
        "per_category": categories,
        "protocol": _protocol_summary(protocol_rows),
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 17},
        "reasoning_allowance_tokens": reasoning_allowance,
    }


def _reasoning_answer_correct(case: ReasoningCase, text: str) -> bool:
    return re.search(
        case.answer_regex,
        text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ) is not None


def _select_reasoning_cases(suite: str) -> tuple[ReasoningCase, ...]:
    if suite == "full":
        return REASONING_CASES
    by_skill: dict[str, list[ReasoningCase]] = defaultdict(list)
    for case in REASONING_CASES:
        by_skill[case.skill].append(case)
    return tuple(
        case
        for skill in ("INF", "DED", "REL", "CSP", "IND", "ABD", "MAG")
        for case in by_skill[skill][:2]
    )


def _one_reasoning_generation(
    model: torch.nn.Module,
    *,
    case: ReasoningCase,
    encoder: Any,
    decode,
    max_seq_len: int,
    precision: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
    rsft: bool,
) -> dict[str, object]:
    template = GPT2ChatTemplate(
        maximum_context_tokens=max_seq_len,
        maximum_assistant_tokens=max_seq_len,
    )
    prompt_ids = template.encode_generation_prompt(
        (ChatMessage("user", case.prompt),),
        encoder,
    )
    budget = min(max_new_tokens, max_seq_len - len(prompt_ids))
    generated = sample_token_ids(
        model,
        prompt_ids,
        max_new_tokens=budget,
        max_seq_len=max_seq_len,
        eos_token_id=EOS_TOKEN_ID,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        precision=precision,
    )
    if rsft:
        protocol = _protocol_fields(generated, encoder)
        answer = str(protocol["answer_text"])
        answer_correct = _reasoning_answer_correct(case, answer)
        return {
            "answer": answer,
            "correct": answer_correct,
            "strict_correct": bool(protocol["well_formed"]) and answer_correct,
            "protocol": {
                key: value
                for key, value in protocol.items()
                if not key.endswith("_token_ids")
            },
            "generated_token_ids": generated,
        }
    terminated = bool(generated and generated[-1] == EOS_TOKEN_ID)
    body = generated[:-1] if terminated else generated
    answer = decode(body).strip()
    return {
        "answer": answer,
        "correct": _reasoning_answer_correct(case, answer),
        "terminated_with_eos": terminated,
        "generated_token_ids": generated,
    }


def _reasoning_suite(
    model: torch.nn.Module,
    *,
    encoder: Any,
    decode,
    max_seq_len: int,
    precision: str,
    suite: str,
    sampled_responses: int,
    max_new_tokens: int,
    rsft: bool,
) -> dict[str, object]:
    cases = _select_reasoning_cases(suite)
    greedy_rows: list[dict[str, object]] = []
    sampled_rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        greedy = _one_reasoning_generation(
            model,
            case=case,
            encoder=encoder,
            decode=decode,
            max_seq_len=max_seq_len,
            precision=precision,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=17 + case_index * 1000,
            max_new_tokens=max_new_tokens,
            rsft=rsft,
        )
        greedy_rows.append(
            {
                "name": case.name,
                "skill": case.skill,
                "level": case.level,
                "prompt": case.prompt,
                **greedy,
            }
        )
        for sample_index in range(sampled_responses):
            sampled = _one_reasoning_generation(
                model,
                case=case,
                encoder=encoder,
                decode=decode,
                max_seq_len=max_seq_len,
                precision=precision,
                temperature=0.6,
                top_p=0.95,
                top_k=0,
                seed=17 + case_index * 1000 + sample_index,
                max_new_tokens=max_new_tokens,
                rsft=rsft,
            )
            sampled_rows.append(
                {
                    "name": case.name,
                    "skill": case.skill,
                    "level": case.level,
                    "sample": sample_index + 1,
                    "prompt": case.prompt,
                    **sampled,
                }
            )

    def grouped_accuracy(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for skill in sorted({str(row["skill"]) for row in rows}):
            selected = [row for row in rows if row["skill"] == skill]
            result[skill] = {
                "runs": len(selected),
                "accuracy": mean(float(bool(row["correct"])) for row in selected),
            }
        return result

    result: dict[str, object] = {
        "schema": "small-llm-rsft-novel-reasoning-v1",
        "case_count": len(cases),
        "greedy": {
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "samples_per_problem": 1,
            },
            "accuracy": mean(float(bool(row["correct"])) for row in greedy_rows),
            "per_skill": grouped_accuracy(greedy_rows),
            "cases": greedy_rows,
        },
        "sampled_pass_at_1": {
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 0,
                "samples_per_problem": sampled_responses,
            },
            "estimate": mean(float(bool(row["correct"])) for row in sampled_rows),
            "per_skill": grouped_accuracy(sampled_rows),
            "runs": sampled_rows,
        },
        "max_new_tokens": max_new_tokens,
    }
    if rsft:
        greedy_protocol = [
            row["protocol"]
            for row in greedy_rows
            if isinstance(row.get("protocol"), Mapping)
        ]
        sampled_protocol = [
            row["protocol"]
            for row in sampled_rows
            if isinstance(row.get("protocol"), Mapping)
        ]
        result["protocol_greedy"] = _protocol_summary(greedy_protocol)
        result["protocol_sampled"] = _protocol_summary(sampled_protocol)
    return result


def _numeric_delta(left: object, right: object) -> float | None:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return None
    return float(right) - float(left)


def _summary_deltas(
    s0: Mapping[str, object],
    rsft: Mapping[str, object],
) -> dict[str, object]:
    s0_eval = s0["eval_core_v1"]
    r_eval = rsft["eval_core_v1"]
    s0_reason = s0["novel_reasoning"]
    r_reason = rsft["novel_reasoning"]
    assert isinstance(s0_eval, Mapping) and isinstance(r_eval, Mapping)
    assert isinstance(s0_reason, Mapping) and isinstance(r_reason, Mapping)
    s0_g = s0_reason["greedy"]
    r_g = r_reason["greedy"]
    s0_p = s0_reason["sampled_pass_at_1"]
    r_p = r_reason["sampled_pass_at_1"]
    assert isinstance(s0_g, Mapping) and isinstance(r_g, Mapping)
    assert isinstance(s0_p, Mapping) and isinstance(r_p, Mapping)
    s0_behavior = s0.get("instruction_behavior")
    r_behavior = rsft.get("instruction_behavior")
    s0_behavior_summary = s0_behavior.get("summary") if isinstance(s0_behavior, Mapping) else {}
    r_behavior_summary = r_behavior.get("summary") if isinstance(r_behavior, Mapping) else {}
    s0_validation = s0.get("s0_validation")
    r_validation = rsft.get("s0_validation")
    s0_top = s0_eval.get("top_k_accuracy")
    r_top = r_eval.get("top_k_accuracy")
    s0_cal = s0_eval.get("calibration")
    r_cal = r_eval.get("calibration")
    return {
        "eval_core_v1": {
            "loss": _numeric_delta(s0_eval.get("loss"), r_eval.get("loss")),
            "perplexity": _numeric_delta(s0_eval.get("perplexity"), r_eval.get("perplexity")),
            "bits_per_byte": _numeric_delta(s0_eval.get("bits_per_byte"), r_eval.get("bits_per_byte")),
            "top_1_accuracy": _numeric_delta(
                s0_top.get("1") if isinstance(s0_top, Mapping) else None,
                r_top.get("1") if isinstance(r_top, Mapping) else None,
            ),
            "top_5_accuracy": _numeric_delta(
                s0_top.get("5") if isinstance(s0_top, Mapping) else None,
                r_top.get("5") if isinstance(r_top, Mapping) else None,
            ),
            "top_10_accuracy": _numeric_delta(
                s0_top.get("10") if isinstance(s0_top, Mapping) else None,
                r_top.get("10") if isinstance(r_top, Mapping) else None,
            ),
            "calibration_ece": _numeric_delta(
                s0_cal.get("ece") if isinstance(s0_cal, Mapping) else None,
                r_cal.get("ece") if isinstance(r_cal, Mapping) else None,
            ),
        },
        "s0_retention": {
            "validation_loss": _numeric_delta(
                s0_validation.get("loss") if isinstance(s0_validation, Mapping) else None,
                r_validation.get("loss") if isinstance(r_validation, Mapping) else None,
            ),
        },
        "instruction_behavior": {
            "pass_rate": _numeric_delta(
                s0_behavior_summary.get("pass_rate") if isinstance(s0_behavior_summary, Mapping) else None,
                r_behavior_summary.get("pass_rate") if isinstance(r_behavior_summary, Mapping) else None,
            ),
        },
        "novel_reasoning": {
            "greedy_accuracy": _numeric_delta(s0_g.get("accuracy"), r_g.get("accuracy")),
            "sampled_pass_at_1": _numeric_delta(s0_p.get("estimate"), r_p.get("estimate")),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-bundle", type=Path, required=True)
    parser.add_argument("--rsft-bundle", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--validation-blocks", type=_positive_int, default=32)
    parser.add_argument("--test-blocks", type=_positive_int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--reasoning-samples",
        type=_positive_int,
        default=DEFAULT_REASONING_SAMPLES,
    )
    parser.add_argument(
        "--reasoning-max-new-tokens",
        type=_positive_int,
        default=DEFAULT_REASONING_MAX_NEW_TOKENS,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--s0-checkpoint-dir", type=Path)
    parser.add_argument("--s0-repo-id")
    parser.add_argument("--s0-run-id", default="100m-2b-sft-s0-001")
    parser.add_argument("--s0-pointer", choices=("best", "latest"), default="latest")
    parser.add_argument("--rsft-checkpoint-dir", type=Path)
    parser.add_argument("--rsft-repo-id")
    parser.add_argument("--rsft-run-id", default="100m-2b-rsft-r0-12306-001")
    parser.add_argument("--rsft-pointer", choices=("best", "latest"), default="latest")
    return parser


def _resolve_s0(
    args: argparse.Namespace,
    *,
    token: str | None,
    temporary: Path,
    device: torch.device,
):
    if args.s0_checkpoint_dir:
        root = args.s0_checkpoint_dir.resolve()
        transport = {"transport": "local", "path": str(root)}
    else:
        if not args.s0_repo_id:
            raise RuntimeError(
                "S0 repository ID is required when --s0-checkpoint-dir is not used"
            )
        root, remote = download_parent_checkpoint(
            repo_id=args.s0_repo_id,
            run_id=args.s0_run_id,
            pointer=args.s0_pointer,
            token=token,
            destination=temporary / "s0",
        )
        transport = {"transport": "remote", **remote}
    checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    pipeline = checkpoint.get("pipeline_state")
    identity = pipeline.get("sft_identity") if isinstance(pipeline, Mapping) else None
    if not isinstance(identity, Mapping) or identity.get("stage") != "sft_s0":
        raise RuntimeError("selected S0 checkpoint does not declare pipeline_state.sft_identity.stage=sft_s0")
    model, config, checkpoint_identity = load_verified_native_checkpoint(root, device=device)
    return model, config, checkpoint_identity, transport


def _resolve_rsft(
    args: argparse.Namespace,
    *,
    token: str | None,
    temporary: Path,
    device: torch.device,
):
    if args.rsft_checkpoint_dir:
        root = args.rsft_checkpoint_dir.resolve()
        transport = {"transport": "local", "path": str(root)}
    else:
        if not args.rsft_repo_id:
            raise RuntimeError(
                "R-SFT repository ID is required when --rsft-checkpoint-dir is not used"
            )
        from trainer.post_pretraining_prompt_suite import download_verified_checkpoint

        root, remote = download_verified_checkpoint(
            repo_id=args.rsft_repo_id,
            run_id=args.rsft_run_id,
            token=token,
            revision=None,
            pointer_name=args.rsft_pointer,
            destination=temporary / "rsft",
        )
        transport = {"transport": "remote", **remote}
    model, config, consumed, reasoning_spec = chat_runtime._load_completed_checkpoint(
        root,
        device=device,
        stage="r-sft",
    )
    checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint_identity = {
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "consumed_tokens": consumed,
        "semantic_vocab_size": config.semantic_vocab_size,
        "pipeline_state": checkpoint.get("pipeline_state"),
    }
    return model, config, checkpoint_identity, transport, reasoning_spec


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s0_bundle = args.s0_bundle.resolve()
    rsft_bundle = args.rsft_bundle.resolve()
    s0_bundle_report = verify_bundle(s0_bundle)
    rsft_bundle_report = verify_bundle(rsft_bundle)
    eval_dir = ensure_eval_core(args.eval_dir.resolve())
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    token = os.environ.get(args.token_env)

    with tempfile.TemporaryDirectory(prefix="small-llm-rsft-qualification-") as temporary_dir:
        temporary = Path(temporary_dir)
        print("Scoring S0 parent checkpoint...", flush=True)
        s0_model, s0_config, s0_identity, s0_transport = _resolve_s0(
            args,
            token=token,
            temporary=temporary,
            device=device,
        )
        normal_encoder = TiktokenGPT2Encoder()
        normal_encoding = _normal_encoding()
        s0_greedy = run_prompt_cases(
            s0_model,
            model_max_seq_len=s0_config.max_seq_len,
            precision=precision,
            questions_only=False,
            max_cases=8 if args.suite == "fast" else None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=17,
            samples_per_prompt=1,
            max_new_tokens=32,
        )
        s0_result: dict[str, object] = {
            "eval_core_v1": evaluate_split(
                s0_model,
                eval_dir=eval_dir,
                suite=args.suite,
                precision=precision,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
            ),
            "s0_validation": _masked_split(
                s0_model,
                precision=precision,
                bundle_root=s0_bundle,
                split="validation",
                maximum_blocks=args.validation_blocks,
                microbatch_size=max(1, args.batch_size),
            ),
            "s0_test": (
                _masked_split(
                    s0_model,
                    precision=precision,
                    bundle_root=s0_bundle,
                    split="test",
                    maximum_blocks=args.test_blocks,
                    microbatch_size=max(1, args.batch_size),
                )
                if args.suite == "full"
                else None
            ),
            "instruction_behavior": evaluate_behavior(
                s0_model,
                precision=precision,
                max_seq_len=s0_config.max_seq_len,
                max_cases=16 if args.suite == "fast" else None,
            ),
            "base_qualitative_prompts": s0_greedy,
            "qualitative_greedy_32": s0_greedy,
            "qualitative_sampled": run_prompt_cases(
                s0_model,
                model_max_seq_len=s0_config.max_seq_len,
                precision=precision,
                questions_only=False,
                max_cases=8 if args.suite == "fast" else None,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                seed=17,
                samples_per_prompt=1,
                max_new_tokens=None,
            ),
            "novel_reasoning": _reasoning_suite(
                s0_model,
                encoder=normal_encoder,
                decode=lambda ids: normal_encoding.decode(list(ids)),
                max_seq_len=s0_config.max_seq_len,
                precision=precision,
                suite=args.suite,
                sampled_responses=args.reasoning_samples,
                max_new_tokens=args.reasoning_max_new_tokens,
                rsft=False,
            ),
        }
        del s0_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("Scoring R-SFT checkpoint...", flush=True)
        rsft_model, rsft_config, rsft_identity, rsft_transport, reasoning_spec = _resolve_rsft(
            args,
            token=token,
            temporary=temporary,
            device=device,
        )
        _assert_transition_compatible(s0_config, rsft_config)
        tokenizer_module = chat_runtime._load_rsft_tokenizer_module()
        rsft_encoder = tokenizer_module.ReasoningGPT2Encoder(
            reasoning_spec,
            base_encoding=normal_encoding,
        )
        rsft_greedy = _base_prompts_rsft(
            rsft_model,
            rsft_encoder=rsft_encoder,
            max_seq_len=rsft_config.max_seq_len,
            precision=precision,
            suite=args.suite,
            sampled=False,
        )
        rsft_result: dict[str, object] = {
            "eval_core_v1": evaluate_split(
                rsft_model,
                eval_dir=eval_dir,
                suite=args.suite,
                precision=precision,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
            ),
            "s0_validation": _masked_split(
                rsft_model,
                precision=precision,
                bundle_root=s0_bundle,
                split="validation",
                maximum_blocks=args.validation_blocks,
                microbatch_size=max(1, args.batch_size),
            ),
            "s0_test": (
                _masked_split(
                    rsft_model,
                    precision=precision,
                    bundle_root=s0_bundle,
                    split="test",
                    maximum_blocks=args.test_blocks,
                    microbatch_size=max(1, args.batch_size),
                )
                if args.suite == "full"
                else None
            ),
            "rsft_validation": _masked_split(
                rsft_model,
                precision=precision,
                bundle_root=rsft_bundle,
                split="validation",
                maximum_blocks=args.validation_blocks,
                microbatch_size=max(1, args.batch_size),
            ),
            "rsft_test": (
                _masked_split(
                    rsft_model,
                    precision=precision,
                    bundle_root=rsft_bundle,
                    split="test",
                    maximum_blocks=args.test_blocks,
                    microbatch_size=max(1, args.batch_size),
                )
                if args.suite == "full"
                else None
            ),
            "instruction_behavior": _evaluate_rsft_behavior(
                rsft_model,
                rsft_encoder=rsft_encoder,
                precision=precision,
                max_seq_len=rsft_config.max_seq_len,
                suite=args.suite,
                reasoning_allowance=max(0, args.reasoning_max_new_tokens - 64),
            ),
            "base_qualitative_prompts": rsft_greedy,
            "qualitative_greedy_32": rsft_greedy,
            "qualitative_sampled": _base_prompts_rsft(
                rsft_model,
                rsft_encoder=rsft_encoder,
                max_seq_len=rsft_config.max_seq_len,
                precision=precision,
                suite=args.suite,
                sampled=True,
            ),
            "novel_reasoning": _reasoning_suite(
                rsft_model,
                encoder=rsft_encoder,
                decode=rsft_encoder.decode,
                max_seq_len=rsft_config.max_seq_len,
                precision=precision,
                suite=args.suite,
                sampled_responses=args.reasoning_samples,
                max_new_tokens=args.reasoning_max_new_tokens,
                rsft=True,
            ),
        }
        del rsft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report: dict[str, object] = {
        "schema": "small-llm-post-rsft-qualification",
        "schema_version": 1,
        "suite": args.suite,
        "selection_policy": {
            "single_master_score": False,
            "interpretation": (
                "inspect reasoning acquisition, instruction behavior, protocol health, "
                "and S0/base retention together"
            ),
            "cot_faithfulness_claim": False,
        },
        "sampling_contracts": {
            "historical_greedy_32": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 32,
                "seed": 17,
            },
            "standard_sampled_native": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 17,
                "budget": "native prompt budget",
            },
            "reasoning_greedy": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": args.reasoning_max_new_tokens,
                "seed": 17,
            },
            "reasoning_pass_at_1": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 0,
                "responses_per_problem": args.reasoning_samples,
                "max_new_tokens": args.reasoning_max_new_tokens,
                "seed": 17,
            },
        },
        "bundles": {
            "s0": s0_bundle_report,
            "rsft": rsft_bundle_report,
        },
        "s0": {
            "checkpoint": s0_identity,
            "transport": s0_transport,
            "scorecard": s0_result,
        },
        "rsft": {
            "checkpoint": rsft_identity,
            "transport": rsft_transport,
            "scorecard": rsft_result,
        },
        "deltas_rsft_minus_s0": _summary_deltas(s0_result, rsft_result),
        "novel_reasoning_suite_identity": canonical_hash(
            [asdict(case) for case in REASONING_CASES]
        ),
    }
    report["report_sha256"] = canonical_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote R-SFT qualification report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
