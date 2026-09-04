"""Evaluation-v2 additions for pretrained Small-LLM checkpoints."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import re
from statistics import mean
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from trainer.eval_generation import GenerationRequest, sample_token_ids_batched

try:
    from lm_eval.api.model import LM as _HarnessBase
except Exception:  # pragma: no cover
    _HarnessBase = object

L20_TASKS = ("arc_challenge", "arc_easy", "hellaswag", "lambada_openai", "piqa", "winogrande")
L20_MAX_BATCH_SIZE = 16
L20_MAX_BATCH_TOKENS = 8_192
BASE_PROMPT_BATCH_SIZE = 16


@dataclass(frozen=True, slots=True)
class BasePromptCase:
    name: str
    family: str
    prompt: str
    max_new_tokens: int
    answer: str | None = None
    regex: str | None = None
    qualitative: bool = False


def _base_items() -> list[tuple[str, str, str, str]]:
    facts = [("capital_france", "Question: What is the capital of France?\nAnswer:", "Paris"), ("largest_planet", "Question: What is the largest planet?\nAnswer:", "Jupiter"), ("red_planet", "Question: Which planet is called the Red Planet?\nAnswer:", "Mars"), ("hamlet", "Question: Who wrote Hamlet?\nAnswer:", "Shakespeare"), ("freezing", "Question: At what Celsius temperature does water freeze?\nAnswer:", "0")]
    items: list[tuple[str, str, str, str]] = []
    for i in range(20):
        n, p, a = facts[i % len(facts)]; items.append((f"factual_{i:02d}_{n}", "factual", p, a))
    pairs = [(7,8), (15,27), (18,5), (6,7), (11,12)]
    for i in range(20):
        a, b = pairs[i % len(pairs)]
        if i % 2 == 0:
            prompt, ans = f"Question: What is {a} multiplied by {b}?\nAnswer:", str(a*b)
        else:
            prompt, ans = f"Question: What is {a} plus {b}?\nAnswer:", str(a+b)
        items.append((f"arithmetic_{i:02d}", "arithmetic", prompt, ans))
    extracts = [("third", "List: apple | pear | mango | grape\nThird item:", "mango"), ("order", "Order number: 4187; status: shipped.\nOrder number:", "4187"), ("invoice", "invoice INV-2048 is overdue.\nInvoice ID:", "INV-2048"), ("city", "City=Lisbon; Country=Portugal.\nCity:", "Lisbon"), ("date", "Meeting date: 2026-09-14 at noon.\nDate:", "2026-09-14")]
    for i in range(20):
        n, p, a = extracts[i % len(extracts)]; items.append((f"extraction_{i:02d}_{n}", "extraction", p, a))
    classes = [("Text: I loved every minute of the film.\nSentiment:", "positive"), ("Text: The plot was tedious and predictable.\nSentiment:", "negative"), ("cat\nClass: animal or object?", "animal"), ("chair\nClass: animal or object?", "object"), ("14\nClass: even or odd?", "even")]
    for i in range(20):
        p, a = classes[i % len(classes)]; items.append((f"classification_{i:02d}", "classification", p, a))
    transforms = [("blue -> uppercase:", "BLUE"), ("SHOUT -> lowercase:", "shout"), ("hello world -> title case:", "Hello World"), ("cat -> reversed:", "tac"), ("dog -> plural:", "dogs")]
    for i in range(20):
        p, a = transforms[i % len(transforms)]; items.append((f"transformation_{i:02d}", "transformation", p, a))
    return items


def base_prompt_cases_v2() -> tuple[BasePromptCase, ...]:
    scored = [BasePromptCase(name, fam, prompt, 48, answer, re.escape(answer)) for name, fam, prompt, answer in _base_items()]
    qualitative_prompts = ["The rain had stopped before dawn, leaving the streets covered in ", "Water can exist as a solid, liquid, or gas. The transition to vapor occurs when ", "The Roman Republic was a period of ancient Roman civilization that began after ", "Alice: Did you close the window?\nBen: I thought you had closed it.\nAlice:", "France | Paris\nItaly | Rome\nGermany |"]
    qualitative = [BasePromptCase(f"qualitative_{i:02d}", "qualitative", qualitative_prompts[i % len(qualitative_prompts)], 128, qualitative=True) for i in range(20)]
    return tuple(scored + qualitative)


BASE_PROMPT_CASES_V2 = base_prompt_cases_v2()


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError(f"{precision} evaluation requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _verdict(case: BasePromptCase, text: str) -> dict[str, object]:
    if case.qualitative:
        return {"scored": False, "passed": None, "checks": {}}
    stripped = text.strip()
    checks = {"non_empty": bool(stripped), "answer_present": case.answer.casefold() in stripped.casefold() if case.answer else True, "regex": re.search(case.regex or "", stripped) is not None if case.regex else True}
    return {"scored": True, "passed": all(checks.values()), "checks": checks}


def _run_prompt_view(model: nn.Module, *, model_max_seq_len: int, precision: str, suite: str, temperature: float, top_p: float, top_k: int, seed: int) -> list[dict[str, object]]:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2"); cases = BASE_PROMPT_CASES_V2 if suite == "full" else BASE_PROMPT_CASES_V2[:30]
    prompt_ids = [enc.encode(case.prompt, disallowed_special=()) for case in cases]
    requests = [GenerationRequest(tuple(ids), min(case.max_new_tokens, model_max_seq_len - len(ids)), seed + index * 1000) for index, (case, ids) in enumerate(zip(cases, prompt_ids, strict=True))]
    view = "greedy" if temperature == 0.0 else "sampled"
    generated_rows = sample_token_ids_batched(model, requests, max_seq_len=model_max_seq_len, eos_token_id=enc.eot_token, temperature=temperature, top_p=top_p, top_k=top_k, precision=precision, batch_size=BASE_PROMPT_BATCH_SIZE, progress_label=f"base-prompts-v2/{view}")
    rows = []
    for case, request, generated in zip(cases, requests, generated_rows, strict=True):
        terminated = bool(generated and generated[-1] == enc.eot_token); body = generated[:-1] if terminated else generated; text = enc.decode(body)
        rows.append({"name": case.name, "family": case.family, "qualitative": case.qualitative, "prompt": case.prompt, "answer": case.answer, "seed": request.seed, "generated_token_ids": generated, "continuation": text, "response_tokens": len(body), "terminated_with_eos": terminated, **_verdict(case, text)})
    return rows


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scored = [r for r in rows if bool(r.get("scored"))]
    fam = {}
    for f in sorted({str(r["family"]) for r in scored}):
        selected = [r for r in scored if r["family"] == f]
        fam[f] = {"cases": len(selected), "accuracy": mean(float(bool(r["passed"])) for r in selected)}
    return {"cases": len(rows), "scored_cases": len(scored), "qualitative_cases": len([r for r in rows if bool(r.get("qualitative"))]), "accuracy": mean(float(bool(r["passed"])) for r in scored) if scored else math.nan, "by_family": fam, "mean_response_tokens": mean(float(r["response_tokens"]) for r in rows) if rows else math.nan}


def run_base_prompt_suite_v2(model: nn.Module, *, model_max_seq_len: int, precision: str, suite: str) -> dict[str, object]:
    greedy = _run_prompt_view(model, model_max_seq_len=model_max_seq_len, precision=precision, suite=suite, temperature=0.0, top_p=1.0, top_k=0, seed=17)
    sampled = _run_prompt_view(model, model_max_seq_len=model_max_seq_len, precision=precision, suite=suite, temperature=1.0, top_p=1.0, top_k=0, seed=17)
    return {"schema": "small-llm-pretraining-base-prompts-v2", "suite_identity": {"full_scored_cases": 100, "full_qualitative_cases": 20, "budget": "native per prompt"}, "execution": {"length_bucketed": True, "max_batch_size": BASE_PROMPT_BATCH_SIZE}, "greedy": {"sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 17}, "summary": _summary(greedy), "cases": greedy}, "sampled": {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 17}, "summary": _summary(sampled), "cases": sampled}}


class SmallLLMHarnessLM(_HarnessBase):
    def __init__(self, model: nn.Module, *, max_seq_len: int, precision: str, batch_size: int = L20_MAX_BATCH_SIZE, max_batch_tokens: int = L20_MAX_BATCH_TOKENS) -> None:
        try: super().__init__()
        except TypeError: pass
        import tiktoken
        if batch_size <= 0 or max_batch_tokens <= 0:
            raise ValueError("L20 batching limits must be positive")
        self.model = model; self.max_seq_len = max_seq_len; self.precision = precision; self.batch_size = batch_size; self.max_batch_tokens = max_batch_tokens; self.encoding = tiktoken.get_encoding("gpt2"); self._rank = 0; self._world_size = 1; self._device = next(model.parameters()).device
    @property
    def rank(self): return 0
    @property
    def world_size(self): return 1
    @property
    def device(self): return self._device
    @property
    def eot_token_id(self): return int(self.encoding.eot_token)
    @property
    def prefix_token_id(self): return self.eot_token_id
    @property
    def tokenizer_name(self): return "gpt2"
    def set_cache_hook(self, cache_hook): self.cache_hook = cache_hook
    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs): return list(self.encoding.encode(string, disallowed_special=()))
    def _encode_pair(self, context: str, continuation: str):
        if not context: return [self.prefix_token_id], self.tok_encode(continuation)
        whole = self.tok_encode(context + continuation); ctx = self.tok_encode(context); return ctx, whole[len(ctx):]
    def _prepare_pair(self, ctx: Sequence[int], cont: Sequence[int]) -> tuple[list[int], int]:
        full = list(ctx) + list(cont)
        if len(full) > self.max_seq_len: full = full[-self.max_seq_len:]
        label_start = max(0, len(full) - len(cont) - 1)
        return full, label_start
    @torch.inference_mode()
    def _score_prepared_batch(self, prepared: Sequence[tuple[list[int], int]]) -> list[tuple[float, bool]]:
        if not prepared:
            return []
        width = max(max(1, len(full) - 1) for full, _ in prepared)
        input_ids = torch.zeros((len(prepared), width), dtype=torch.long, device=self.device)
        for row, (full, _) in enumerate(prepared):
            if len(full) > 1:
                input_ids[row, : len(full) - 1] = torch.tensor(full[:-1], dtype=torch.long, device=self.device)
        with _autocast(self.device, self.precision):
            logits = self.model(input_ids)
        if isinstance(logits, tuple): logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("L20 model must return [batch, time, vocab] logits")
        results: list[tuple[float, bool]] = []
        for row, (full, label_start) in enumerate(prepared):
            if len(full) <= 1:
                results.append((0.0, True)); continue
            end = len(full) - 1
            selected_logits = logits[row, label_start:end, :].float()
            selected_labels = torch.tensor(full[1:], dtype=torch.long, device=self.device)[label_start:end]
            if selected_labels.numel() == 0:
                results.append((0.0, True)); continue
            lp = F.log_softmax(selected_logits, dim=-1)
            results.append((float(lp.gather(1, selected_labels[:, None]).sum()), bool(torch.eq(selected_logits.argmax(dim=-1), selected_labels).all())))
        return results
    @torch.inference_mode()
    def _score_pair(self, ctx: Sequence[int], cont: Sequence[int]) -> tuple[float, bool]:
        return self._score_prepared_batch((self._prepare_pair(ctx, cont),))[0]
    def loglikelihood(self, requests: list[Any], disable_tqdm: bool = False):
        del disable_tqdm
        encoded = []
        for index, req in enumerate(requests):
            ctx, cont = self._encode_pair(str(req.args[0]), str(req.args[1])); prepared = self._prepare_pair(ctx, cont)
            encoded.append((index, prepared, max(1, len(prepared[0]) - 1)))
        encoded.sort(key=lambda item: (item[2], item[0]))
        batches: list[list[tuple[int, tuple[list[int], int], int]]] = []; current = []; current_width = 0
        for item in encoded:
            candidate_width = max(current_width, item[2]); candidate_count = len(current) + 1
            if current and (candidate_count > self.batch_size or candidate_width * candidate_count > self.max_batch_tokens):
                batches.append(current); current = []; current_width = 0; candidate_width = item[2]
            current.append(item); current_width = max(current_width, candidate_width)
        if current: batches.append(current)
        results: list[tuple[float, bool] | None] = [None] * len(requests); total = len(requests); completed = 0; next_report = 10
        print(f"[L20 likelihood] scoring {total} requests in {len(batches)} length-bucketed batches (max_batch_size={self.batch_size}, max_batch_tokens={self.max_batch_tokens})", flush=True)
        for batch in batches:
            scores = self._score_prepared_batch([item[1] for item in batch])
            for (original_index, _, _), score in zip(batch, scores, strict=True): results[original_index] = score
            completed += len(batch); percent = int(completed * 100 / max(1, total))
            if percent >= next_report or completed == total:
                print(f"[L20 likelihood] {completed}/{total} requests complete ({percent}%)", flush=True); next_report = min(100, ((percent // 10) + 1) * 10)
        if any(result is None for result in results): raise RuntimeError("L20 batched scorer failed to populate every request")
        return [result for result in results if result is not None]
    def loglikelihood_rolling(self, requests: list[Any], disable_tqdm: bool = False): raise RuntimeError("rolling likelihood disabled for L20")
    def generate_until(self, requests: list[Any], disable_tqdm: bool = False): raise RuntimeError("generation disabled for L20")
    def apply_chat_template(self, chat_history, add_generation_prompt=True): raise RuntimeError("chat templates disabled for L20")
    def chat_template(self, chat_template=False): return None


def _metric(row: Mapping[str, object]) -> float | None:
    for key in ("acc_norm", "acc", "exact_match"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def run_l20_conditional_likelihood(model: nn.Module, *, model_max_seq_len: int, precision: str, suite: str) -> dict[str, object]:
    try: import lm_eval
    except ImportError as error: raise RuntimeError("install evaluation dependencies with: python -m pip install -r requirements-eval.txt") from error
    adapter = SmallLLMHarnessLM(model, max_seq_len=model_max_seq_len, precision=precision)
    result = lm_eval.simple_evaluate(model=adapter, tasks=list(L20_TASKS), num_fewshot=0, batch_size=L20_MAX_BATCH_SIZE, limit=100 if suite == "fast" else None, bootstrap_iters=0, log_samples=False, random_seed=17, numpy_random_seed=17, torch_random_seed=17, fewshot_random_seed=17)
    raw = result.get("results", {}) if isinstance(result, Mapping) else {}; rows = {}; values = []
    for task in L20_TASKS:
        row = raw.get(task, {}) if isinstance(raw, Mapping) else {}; metric = _metric(row) if isinstance(row, Mapping) else None
        if metric is not None: values.append(metric)
        rows[task] = {"headline_metric": metric, "raw": dict(row) if isinstance(row, Mapping) else {}}
    return {"schema": "small-llm-l20-conditional-likelihood-v1", "harness": "lm-evaluation-harness==0.4.12", "tasks": list(L20_TASKS), "limit_per_task": 100 if suite == "fast" else None, "mean_6": mean(values) if len(values) == 6 else math.nan, "task_results": rows, "request_policy": {"loglikelihood": True, "rolling_loglikelihood": False, "generation": False}, "execution": {"length_bucketed": True, "max_batch_size": L20_MAX_BATCH_SIZE, "max_batch_tokens": L20_MAX_BATCH_TOKENS, "world_size": adapter.world_size}}


__all__ = ["BASE_PROMPT_CASES_V2", "BasePromptCase", "L20_TASKS", "run_base_prompt_suite_v2", "run_l20_conditional_likelihood", "SmallLLMHarnessLM"]
