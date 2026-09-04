"""Mechanically scored SFT Behavior v2 benchmark.

This additive module defines the frozen v2 shape approved by ADR 0140:
180 semantic tasks, four paired L0-L3 variants, 720 cases, diagnostic and
held-out qualification partitions, conditional compliance, and paired deltas.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
from statistics import mean
from typing import Iterable, Mapping, Sequence

from torch import nn

from trainer.eval_generation import DEFAULT_GENERATION_BATCH_SIZE, GenerationRequest, sample_token_ids_batched

from .schema import ChatMessage
from .template import GPT2ChatTemplate, TiktokenGPT2Encoder

LEVELS = ("L0", "L1", "L2", "L3")
FAMILIES = ("direct_qa", "arithmetic_reasoning", "extraction", "classification", "transformation", "short_summarization")
SAMPLED_ROBUSTNESS_SEEDS = (17, 18, 19)
ROLE_LEAK_PATTERN = re.compile(r"(?:^|\n)(?:System|User|Assistant):", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BehaviorV2Task:
    semantic_id: str
    family: str
    prompt: str
    answer: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BehaviorV2Case:
    case_id: str
    semantic_id: str
    family: str
    level: str
    partition: str
    messages: tuple[ChatMessage, ...]
    answer: str
    aliases: tuple[str, ...] = ()
    max_new_tokens: int = 64


def _task_rows() -> tuple[BehaviorV2Task, ...]:
    facts = [
        ("capital_france", "What is the capital of France?", "Paris"), ("largest_planet", "What is the largest planet in the Solar System?", "Jupiter"),
        ("brazil_language", "What is the main language spoken in Brazil?", "Portuguese"), ("red_planet", "Which planet is called the Red Planet?", "Mars"),
        ("hamlet", "Who wrote Hamlet?", "Shakespeare"), ("largest_ocean", "What is the largest ocean on Earth?", "Pacific Ocean"),
        ("japan_currency", "What is the currency of Japan?", "yen"), ("blood_pump", "Which organ pumps blood?", "heart"),
        ("leap_year", "How many days are in a leap year?", "366"), ("photosynthesis", "What process do plants use to convert light to chemical energy?", "photosynthesis"),
        ("italy_capital", "What is the capital of Italy?", "Rome"), ("spain_capital", "What is the capital of Spain?", "Madrid"),
        ("uk_capital", "What is the capital of the United Kingdom?", "London"), ("canada_capital", "What is the capital of Canada?", "Ottawa"),
        ("water_formula", "What is the chemical formula for water?", "H2O"), ("triangle_sides", "How many sides does a triangle have?", "3"),
        ("week_days", "How many days are in one week?", "7"), ("moon", "What is Earth's natural satellite called?", "Moon"),
        ("oxygen", "What gas do humans need to breathe?", "oxygen"), ("sun", "What star is at the center of the Solar System?", "Sun"),
        ("mexico_capital", "What is the capital of Mexico?", "Mexico City"), ("egypt_continent", "On what continent is Egypt?", "Africa"),
        ("first_month", "What is the first month of the year?", "January"), ("opposite_north", "What direction is opposite north?", "south"),
        ("caterpillar", "What animal does a caterpillar become?", "butterfly"), ("hearing", "Which sense uses the ears?", "hearing"),
        ("boiling", "At what Celsius temperature does water boil at sea level?", "100"), ("freezing", "At what Celsius temperature does water freeze?", "0"),
        ("whale", "Is a whale a mammal?", "yes"), ("red_primary", "Is red a common primary color in art?", "yes"),
    ]
    tasks: list[BehaviorV2Task] = [BehaviorV2Task(f"direct_{name}", "direct_qa", prompt, answer) for name, prompt, answer in facts]
    pairs = [(7, 8), (15, 27), (18, 5), (6, 7), (11, 12), (100, 37), (25, 4), (81, 9), (13, 13), (17, 9)]
    for i in range(30):
        a, b = pairs[i % len(pairs)]
        if i % 3 == 0:
            prompt, answer = f"What is {a} multiplied by {b}?", str(a * b)
        elif i % 3 == 1:
            prompt, answer = f"What is {a} plus {b}?", str(a + b)
        else:
            high, low = max(a, b), min(a, b); prompt, answer = f"What is {high} minus {low}?", str(high - low)
        tasks.append(BehaviorV2Task(f"arith_{i:02d}", "arithmetic_reasoning", prompt, answer))
    values = [("order", "Order number: 4187; status: shipped.", "4187"), ("third", "apple | pear | mango | grape", "mango"), ("invoice", "invoice INV-2048 is overdue", "INV-2048"), ("code", "The temporary code is 731", "731"), ("city", "City=Lisbon; Country=Portugal", "Lisbon")]
    for i in range(30):
        name, source, answer = values[i % len(values)]
        tasks.append(BehaviorV2Task(f"extract_{i:02d}_{name}", "extraction", f"Extract the target value from: {source}", answer))
    classes = [("I loved every minute of the film.", "positive"), ("The plot was tedious and predictable.", "negative"), ("cat", "animal"), ("chair", "object"), ("14", "even"), ("17", "odd")]
    for i in range(30):
        source, answer = classes[i % len(classes)]
        tasks.append(BehaviorV2Task(f"class_{i:02d}", "classification", f"Classify this item with the most specific label: {source}", answer))
    transforms = [("uppercase of blue", "BLUE"), ("lowercase of SHOUT", "shout"), ("title case of hello world", "Hello World"), ("reverse of cat", "tac"), ("plural of dog", "dogs")]
    for i in range(30):
        prompt, answer = transforms[i % len(transforms)]
        tasks.append(BehaviorV2Task(f"transform_{i:02d}", "transformation", f"Return the {prompt}.", answer))
    summaries = [("Rain fell all night and the streets were wet by morning.", "rain"), ("The train arrived thirty minutes late after a signal problem.", "delay"), ("Friends gathered with cake and candles to celebrate Mia turning ten.", "birthday"), ("Smoke filled the kitchen after the pan caught fire.", "fire"), ("The home team scored last and won the match.", "win")]
    for i in range(30):
        text, answer = summaries[i % len(summaries)]
        tasks.append(BehaviorV2Task(f"summary_{i:02d}", "short_summarization", f"Summarize in one word: {text}", answer))
    return tuple(tasks)


def _case_text(task: BehaviorV2Task, level: str) -> str:
    if level == "L0":
        return f"Answer this question.\n\n{task.prompt}"
    if level == "L1":
        return f"Return only the answer, with no explanation.\n\n{task.prompt}"
    if level == "L2":
        return "Return exactly one line in this format and nothing else:\nAnswer: <answer>\n\nTask: " + task.prompt
    return "Return exactly two lines and nothing else. The first line must be `Answer: <answer>` and the second line must be `Done: yes`.\n\nTask: " + task.prompt


def _cases(tasks: Sequence[BehaviorV2Task]) -> tuple[BehaviorV2Case, ...]:
    seen: dict[str, int] = defaultdict(int); rows: list[BehaviorV2Case] = []
    for task in tasks:
        index = seen[task.family]; seen[task.family] += 1
        partition = "diagnostic" if index < 20 else "qualification"
        for level in LEVELS:
            rows.append(BehaviorV2Case(f"{task.semantic_id}:{level}", task.semantic_id, task.family, level, partition, (ChatMessage("user", _case_text(task, level)),), task.answer, task.aliases))
    return tuple(rows)


BEHAVIOR_V2_TASKS = _task_rows()
BEHAVIOR_V2_CASES = _cases(BEHAVIOR_V2_TASKS)


def _answer_options(case: BehaviorV2Case) -> tuple[str, ...]:
    return (case.answer, *case.aliases)


def _contains_answer(text: str, case: BehaviorV2Case) -> bool:
    folded = text.casefold()
    return any(option.casefold() in folded for option in _answer_options(case))


def _exact_options(case: BehaviorV2Case) -> tuple[str, ...]:
    if case.level == "L1":
        return _answer_options(case)
    if case.level == "L2":
        return tuple(f"Answer: {x}" for x in _answer_options(case))
    if case.level == "L3":
        return tuple(f"Answer: {x}\nDone: yes" for x in _answer_options(case))
    return ()


def _ngram_repetition(ids: Sequence[int], n: int = 3) -> float:
    values = tuple(int(x) for x in ids)
    if len(values) < n:
        return 0.0
    grams = [values[i:i+n] for i in range(len(values)-n+1)]
    return 1.0 - len(set(grams)) / len(grams)


def verify_behavior_v2_response(case: BehaviorV2Case, *, text: str, response_token_ids: Sequence[int], terminated_with_eos: bool) -> dict[str, object]:
    stripped = text.strip(); checks: dict[str, bool] = {"content_correct": _contains_answer(stripped, case), "non_empty": bool(stripped), "eos": terminated_with_eos, "no_role_leak": ROLE_LEAK_PATTERN.search(stripped) is None}
    if case.level == "L0":
        checks["format_constraints"] = True
    elif case.level == "L1":
        checks["answer_only"] = stripped in _exact_options(case); checks["format_constraints"] = checks["answer_only"]
    elif case.level == "L2":
        checks["one_line"] = "\n" not in stripped; checks["answer_prefix"] = stripped.startswith("Answer: "); checks["exact_format"] = stripped in _exact_options(case); checks["format_constraints"] = checks["one_line"] and checks["answer_prefix"] and checks["exact_format"]
    else:
        lines = stripped.splitlines(); checks["two_lines"] = len(lines) == 2; checks["answer_prefix"] = bool(lines) and lines[0].startswith("Answer: "); checks["done_line"] = len(lines) == 2 and lines[1] == "Done: yes"; checks["exact_format"] = stripped in _exact_options(case); checks["format_constraints"] = checks["two_lines"] and checks["answer_prefix"] and checks["done_line"] and checks["exact_format"]
    role_leak = not checks["no_role_leak"]
    return {"passed": all(checks.values()), "checks": checks, "response_text": text, "response_tokens": len(response_token_ids), "terminated_with_eos": terminated_with_eos, "runaway": not terminated_with_eos, "empty": not bool(stripped), "role_leak": role_leak, "trigram_repetition": _ngram_repetition(response_token_ids)}


def _select(partition: str, max_cases: int | None) -> tuple[BehaviorV2Case, ...]:
    rows = tuple(c for c in BEHAVIOR_V2_CASES if partition == "all" or c.partition == partition)
    return rows[:max_cases] if max_cases is not None else rows


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"cases": 0}
    l0 = {str(r["semantic_id"]): bool(r["checks"]["content_correct"]) for r in rows if r["level"] == "L0"}  # type: ignore[index]
    conditional = [r for r in rows if r["level"] != "L0" and l0.get(str(r["semantic_id"]), False)]
    def group(key: str) -> dict[str, object]:
        out = {}
        for value in sorted({str(r[key]) for r in rows}):
            selected = [r for r in rows if r[key] == value]
            out[value] = {"cases": len(selected), "pass_rate": mean(float(bool(r["passed"])) for r in selected), "content_accuracy": mean(float(bool(r["checks"]["content_correct"])) for r in selected)}  # type: ignore[index]
        return out
    return {"cases": len(rows), "semantic_tasks": len({str(r["semantic_id"]) for r in rows}), "pass_rate": mean(float(bool(r["passed"])) for r in rows), "content_accuracy": mean(float(bool(r["checks"]["content_correct"])) for r in rows), "conditional_compliance": {"eligible_cases": len(conditional), "pass_rate": mean(float(bool(r["passed"])) for r in conditional) if conditional else math.nan}, "by_partition": group("partition"), "by_family": group("family"), "by_level": group("level")}


def _run(model: nn.Module, *, precision: str, max_seq_len: int, cases: Sequence[BehaviorV2Case], temperature: float, top_p: float, top_k: int, seeds: Sequence[int]) -> list[dict[str, object]]:
    import tiktoken
    encoding = tiktoken.get_encoding("gpt2"); encoder = TiktokenGPT2Encoder(); template = GPT2ChatTemplate(maximum_context_tokens=max_seq_len)
    work = []
    for case_index, case in enumerate(cases):
        prompt_ids = template.encode_generation_prompt(case.messages, encoder)
        for sample_index, seed in enumerate(seeds, 1):
            sample_seed = seed + case_index * 1000
            request = GenerationRequest(tuple(prompt_ids), min(case.max_new_tokens, max_seq_len - len(prompt_ids)), sample_seed)
            work.append((case, sample_index, sample_seed, request))
    view = "greedy" if temperature == 0.0 else "sampled"
    generated_rows = sample_token_ids_batched(model, [item[3] for item in work], max_seq_len=max_seq_len, eos_token_id=template.eos_token_id, temperature=temperature, top_p=top_p, top_k=top_k, precision=precision, batch_size=DEFAULT_GENERATION_BATCH_SIZE, progress_label=f"behavior-v2/{view}")
    rows = []
    for (case, sample_index, sample_seed, _), generated in zip(work, generated_rows, strict=True):
        terminated = bool(generated and generated[-1] == template.eos_token_id); body = generated[:-1] if terminated else generated
        verdict = verify_behavior_v2_response(case, text=encoding.decode(body), response_token_ids=body, terminated_with_eos=terminated)
        rows.append({"case_id": case.case_id, "semantic_id": case.semantic_id, "family": case.family, "level": case.level, "partition": case.partition, "sample": sample_index, "seed": sample_seed, "prompt": case.messages[-1].content, "answer": case.answer, **verdict})
    return rows


def evaluate_behavior_v2(model: nn.Module, *, precision: str, max_seq_len: int, partition: str = "all", max_cases: int | None = None, sampled: bool = True) -> dict[str, object]:
    cases = _select(partition, max_cases)
    greedy = _run(model, precision=precision, max_seq_len=max_seq_len, cases=cases, temperature=0.0, top_p=1.0, top_k=0, seeds=(17,))
    sampled_rows = _run(model, precision=precision, max_seq_len=max_seq_len, cases=cases, temperature=1.0, top_p=1.0, top_k=0, seeds=SAMPLED_ROBUSTNESS_SEEDS) if sampled else []
    return {"schema": "small-llm-sft-behavior-v2", "suite_identity": {"semantic_tasks": len(BEHAVIOR_V2_TASKS), "cases": len(BEHAVIOR_V2_CASES), "diagnostic_cases": 480, "qualification_cases": 240, "families": list(FAMILIES), "levels": list(LEVELS)}, "selection": {"partition": partition, "cases": len(cases)}, "execution": {"length_bucketed": True, "max_batch_size": DEFAULT_GENERATION_BATCH_SIZE}, "greedy": {"sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 17}, "summary": _summary(greedy), "cases": greedy}, "sampled_robustness": {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seeds": list(SAMPLED_ROBUSTNESS_SEEDS)}, "summary": _summary(sampled_rows) if sampled_rows else {"cases": 0}, "cases": sampled_rows}}


def _binom_p(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, min(k, n-k)+1)) / (2 ** n))


def paired_behavior_v2_deltas(parent: Mapping[str, object], tuned: Mapping[str, object]) -> dict[str, object]:
    pc = parent.get("greedy", {}).get("cases", []) if isinstance(parent.get("greedy"), Mapping) else []
    tc = tuned.get("greedy", {}).get("cases", []) if isinstance(tuned.get("greedy"), Mapping) else []
    p = {str(r["case_id"]): bool(r["passed"]) for r in pc if isinstance(r, Mapping) and "case_id" in r}; t = {str(r["case_id"]): bool(r["passed"]) for r in tc if isinstance(r, Mapping) and "case_id" in r}
    ids = sorted(set(p) & set(t)); wins = [i for i in ids if not p[i] and t[i]]; losses = [i for i in ids if p[i] and not t[i]]
    return {"paired_cases": len(ids), "wins_sft_over_parent": len(wins), "losses_sft_under_parent": len(losses), "ties": len(ids) - len(wins) - len(losses), "net_wins": len(wins) - len(losses), "mcnemar_exact_two_sided_p": _binom_p(len(wins), len(wins) + len(losses)), "example_wins": wins[:20], "example_losses": losses[:20]}


def behavior_v2_prompt_texts() -> tuple[str, ...]:
    return tuple(m.content for c in BEHAVIOR_V2_CASES for m in c.messages if m.role != "assistant")


__all__ = ["BEHAVIOR_V2_CASES", "BEHAVIOR_V2_TASKS", "BehaviorV2Case", "BehaviorV2Task", "evaluate_behavior_v2", "paired_behavior_v2_deltas", "behavior_v2_prompt_texts", "verify_behavior_v2_response"]
