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
BASE_PROMPT_SET_ID = "base-prompt-v2-unique-120-2026-09-04"


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
    families: dict[str, tuple[tuple[str, str, str], ...]] = {
        "factual": (
            ("capital_france", "Question: What is the capital of France?\nAnswer:", "Paris"),
            ("largest_planet", "Question: What is the largest planet in the Solar System?\nAnswer:", "Jupiter"),
            ("red_planet", "Question: Which planet is commonly called the Red Planet?\nAnswer:", "Mars"),
            ("hamlet_author", "Question: Who wrote the play Hamlet?\nAnswer:", "Shakespeare"),
            ("water_freezing", "Question: At what Celsius temperature does pure water freeze at standard pressure?\nAnswer:", "0"),
            ("largest_ocean", "Question: What is the largest ocean on Earth?\nAnswer:", "Pacific Ocean"),
            ("brazil_language", "Question: What is the main language spoken in Brazil?\nAnswer:", "Portuguese"),
            ("japan_currency", "Question: What is the currency of Japan?\nAnswer:", "yen"),
            ("blood_pump", "Question: Which organ pumps blood around the human body?\nAnswer:", "heart"),
            ("water_formula", "Question: What is the chemical formula for water?\nAnswer:", "H2O"),
            ("continents", "Question: How many continents are commonly recognized?\nAnswer:", "7"),
            ("nineteen_eighty_four_author", "Question: Who wrote the novel 1984?\nAnswer:", "George Orwell"),
            ("capital_italy", "Question: What is the capital of Italy?\nAnswer:", "Rome"),
            ("tallest_land_animal", "Question: What is the tallest living land animal?\nAnswer:", "giraffe"),
            ("plants_absorb", "Question: Which gas do plants absorb from the atmosphere during photosynthesis?\nAnswer:", "carbon dioxide"),
            ("earth_star", "Question: What is the name of the star that Earth orbits?\nAnswer:", "Sun"),
            ("largest_mammal", "Question: What is the largest living mammal?\nAnswer:", "blue whale"),
            ("capital_spain", "Question: What is the capital of Spain?\nAnswer:", "Madrid"),
            ("hardest_natural_substance", "Question: What is the hardest naturally occurring substance?\nAnswer:", "diamond"),
            ("earth_moon", "Question: What is the name of Earth's natural satellite?\nAnswer:", "Moon"),
        ),
        "arithmetic": (
            ("mul_7_8", "Question: What is 7 multiplied by 8?\nAnswer:", "56"),
            ("add_15_27", "Question: What is 15 plus 27?\nAnswer:", "42"),
            ("sub_93_38", "Question: What is 93 minus 38?\nAnswer:", "55"),
            ("div_84_7", "Question: What is 84 divided by 7?\nAnswer:", "12"),
            ("mul_12_9", "Question: What is 12 multiplied by 9?\nAnswer:", "108"),
            ("add_46_35", "Question: What is 46 plus 35?\nAnswer:", "81"),
            ("sub_120_47", "Question: What is 120 minus 47?\nAnswer:", "73"),
            ("div_144_12", "Question: What is 144 divided by 12?\nAnswer:", "12"),
            ("mul_14_6", "Question: What is 14 multiplied by 6?\nAnswer:", "84"),
            ("add_128_64", "Question: What is 128 plus 64?\nAnswer:", "192"),
            ("sub_1000_275", "Question: What is 1000 minus 275?\nAnswer:", "725"),
            ("div_225_15", "Question: What is 225 divided by 15?\nAnswer:", "15"),
            ("mul_17_5", "Question: What is 17 multiplied by 5?\nAnswer:", "85"),
            ("add_303_99", "Question: What is 303 plus 99?\nAnswer:", "402"),
            ("sub_71_29", "Question: What is 71 minus 29?\nAnswer:", "42"),
            ("div_96_8", "Question: What is 96 divided by 8?\nAnswer:", "12"),
            ("mul_25_16", "Question: What is 25 multiplied by 16?\nAnswer:", "400"),
            ("add_234_567", "Question: What is 234 plus 567?\nAnswer:", "801"),
            ("sub_500_123", "Question: What is 500 minus 123?\nAnswer:", "377"),
            ("div_360_9", "Question: What is 360 divided by 9?\nAnswer:", "40"),
        ),
        "extraction": (
            ("third_fruit", "List: apple | pear | mango | grape\nThird item:", "mango"),
            ("order_number", "Order number: 4187; status: shipped.\nOrder number:", "4187"),
            ("invoice_id", "Invoice INV-2048 is overdue.\nInvoice ID:", "INV-2048"),
            ("city", "City=Lisbon; Country=Portugal.\nCity:", "Lisbon"),
            ("meeting_date", "Meeting date: 2026-09-14 at noon.\nDate:", "2026-09-14"),
            ("tracking_code", "Package tracking code ZX-7319; carrier: NorthPost.\nTracking code:", "ZX-7319"),
            ("second_color", "Colors in order: amber, teal, violet, silver.\nSecond color:", "teal"),
            ("username", "Account: username=river_fox; role=editor.\nUsername:", "river_fox"),
            ("temperature", "Sensor report: humidity 41%; temperature 23 C; pressure 1014 hPa.\nTemperature:", "23 C"),
            ("product_code", "Product code P-8821, quantity 6, warehouse B.\nProduct code:", "P-8821"),
            ("surname", "Passenger: Elena Rossi; seat 14A.\nSurname:", "Rossi"),
            ("third_city", "Route: Oslo -> Copenhagen -> Berlin -> Prague.\nThird city:", "Berlin"),
            ("ticket_id", "Support ticket TKT-5902 is marked resolved.\nTicket ID:", "TKT-5902"),
            ("version", "Release notes for version 3.7.2 were published today.\nVersion:", "3.7.2"),
            ("room", "Reservation: guest Malik Chen, room 512, two nights.\nRoom number:", "512"),
            ("isbn", "Book record: title=North Wind; ISBN=978-1-4028-9462-6.\nISBN:", "978-1-4028-9462-6"),
            ("first_planet", "Planets listed: Mercury, Venus, Earth, Mars.\nFirst planet:", "Mercury"),
            ("email_domain", "Contact email: maya@example.org.\nEmail domain:", "example.org"),
            ("batch_code", "Factory log: batch BQ-44; line 3; status passed.\nBatch code:", "BQ-44"),
            ("latitude", "Coordinates: latitude 41.9028, longitude 12.4964.\nLatitude:", "41.9028"),
        ),
        "classification": (
            ("sentiment_positive", "Text: I loved every minute of the film.\nLabel as positive or negative sentiment:", "positive"),
            ("sentiment_negative", "Text: The plot was tedious and predictable.\nLabel as positive or negative sentiment:", "negative"),
            ("cat_animal", "Item: cat\nClassify as animal or object:", "animal"),
            ("chair_object", "Item: chair\nClassify as animal or object:", "object"),
            ("fourteen_even", "Number: 14\nClassify as even or odd:", "even"),
            ("twenty_one_odd", "Number: 21\nClassify as even or odd:", "odd"),
            ("apple_fruit", "Food: apple\nClassify as fruit or vegetable:", "fruit"),
            ("carrot_vegetable", "Food: carrot\nClassify as fruit or vegetable:", "vegetable"),
            ("eagle_bird", "Animal: eagle\nClassify as bird or mammal:", "bird"),
            ("dolphin_mammal", "Animal: dolphin\nClassify as bird or mammal:", "mammal"),
            ("water_liquid", "Substance at room temperature: water\nClassify as solid, liquid, or gas:", "liquid"),
            ("oxygen_gas", "Substance at room temperature: oxygen\nClassify as solid, liquid, or gas:", "gas"),
            ("granite_solid", "Substance at room temperature: granite\nClassify as solid, liquid, or gas:", "solid"),
            ("question_interrogative", "Sentence: Where did you leave the keys?\nClassify as statement or question:", "question"),
            ("statement_declarative", "Sentence: The train arrives at six.\nClassify as statement or question:", "statement"),
            ("python_programming", "Term: Python\nClassify as programming language or planet:", "programming language"),
            ("saturn_planet", "Term: Saturn\nClassify as programming language or planet:", "planet"),
            ("triangle_polygon", "Shape: triangle\nClassify as polygon or circle:", "polygon"),
            ("circle_circle", "Shape: circle\nClassify as polygon or circle:", "circle"),
            ("email_digital", "Message type: email\nClassify as digital or physical mail:", "digital"),
        ),
        "transformation": (
            ("upper_blue", "Transform to uppercase: blue\nResult:", "BLUE"),
            ("lower_shout", "Transform to lowercase: SHOUT\nResult:", "shout"),
            ("title_hello_world", "Transform to title case: hello world\nResult:", "Hello World"),
            ("reverse_cat", "Reverse the characters in: cat\nResult:", "tac"),
            ("plural_dog", "Write the regular plural of: dog\nResult:", "dogs"),
            ("upper_lisbon", "Transform to uppercase: Lisbon\nResult:", "LISBON"),
            ("lower_mixed", "Transform to lowercase: MIXED\nResult:", "mixed"),
            ("title_small_model", "Transform to title case: small model\nResult:", "Small Model"),
            ("reverse_train", "Reverse the characters in: train\nResult:", "niart"),
            ("plural_book", "Write the regular plural of: book\nResult:", "books"),
            ("upper_orange", "Transform to uppercase: orange\nResult:", "ORANGE"),
            ("lower_quiet", "Transform to lowercase: QUIET\nResult:", "quiet"),
            ("title_red_fox", "Transform to title case: red fox\nResult:", "Red Fox"),
            ("reverse_planet", "Reverse the characters in: planet\nResult:", "tenalp"),
            ("plural_car", "Write the regular plural of: car\nResult:", "cars"),
            ("upper_delta", "Transform to uppercase: delta\nResult:", "DELTA"),
            ("lower_window", "Transform to lowercase: WINDOW\nResult:", "window"),
            ("title_open_door", "Transform to title case: open door\nResult:", "Open Door"),
            ("reverse_music", "Reverse the characters in: music\nResult:", "cisum"),
            ("plural_tree", "Write the regular plural of: tree\nResult:", "trees"),
        ),
    }
    items: list[tuple[str, str, str, str]] = []
    for family, rows in families.items():
        if len(rows) != 20:
            raise RuntimeError(f"Base Prompt v2 family {family!r} must contain exactly 20 cases")
        for index, (slug, prompt, answer) in enumerate(rows):
            items.append((f"{family}_{index:02d}_{slug}", family, prompt, answer))
    return items


def _validate_base_prompt_cases(cases: Sequence[BasePromptCase]) -> None:
    if len(cases) != 120:
        raise RuntimeError(f"Base Prompt v2 must contain exactly 120 cases, got {len(cases)}")
    names = [case.name for case in cases]
    prompts = [case.prompt for case in cases]
    if len(set(names)) != len(names):
        raise RuntimeError("Base Prompt v2 contains duplicate case IDs")
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("Base Prompt v2 contains duplicate prompt text")

    scored = [case for case in cases if not case.qualitative]
    qualitative = [case for case in cases if case.qualitative]
    if len(scored) != 100 or len(qualitative) != 20:
        raise RuntimeError("Base Prompt v2 must contain 100 scored and 20 qualitative cases")
    expected_families = {"factual", "arithmetic", "extraction", "classification", "transformation"}
    for family in expected_families:
        count = sum(case.family == family for case in scored)
        if count != 20:
            raise RuntimeError(f"Base Prompt v2 scored family {family!r} must contain 20 cases, got {count}")
    if any(case.family not in expected_families for case in scored):
        raise RuntimeError("Base Prompt v2 contains an unexpected scored family")
    if any(case.answer is None or case.regex is None for case in scored):
        raise RuntimeError("every scored Base Prompt v2 case must define answer and regex")


def base_prompt_cases_v2() -> tuple[BasePromptCase, ...]:
    scored = [
        BasePromptCase(name, family, prompt, 48, answer, re.escape(answer))
        for name, family, prompt, answer in _base_items()
    ]
    qualitative_prompts = (
        "The rain had stopped before dawn, leaving the streets covered in ",
        "Water can exist as a solid, liquid, or gas. The transition to vapor occurs when ",
        "The Roman Republic was a period of ancient Roman civilization that began after ",
        "Alice: Did you close the window?\nBen: I thought you had closed it.\nAlice:",
        "France | Paris\nItaly | Rome\nGermany |",
        "The old radio crackled once, and then a distant voice said, ",
        "Photosynthesis allows green plants to convert light energy into ",
        "At the edge of the forest, the path split in two. Mira chose the left path because ",
        "Doctor: How long have you had the cough?\nPatient: About three days.\nDoctor:",
        "Mercury | planet\nSirius | star\nAndromeda |",
        "The instructions were simple: first rinse the rice, then add water, and finally ",
        "Computer memory stores information that a processor can access. In general, faster memory ",
        "Leo opened the envelope and found no letter inside, only ",
        "Teacher: Why does ice float on water?\nStudent:",
        "2, 4, 8, 16, ",
        "The museum closed at six, but one gallery light remained on because ",
        "Sound travels through air as variations in ",
        "Nora checked the departure board again. Her train was delayed, so she ",
        "oak | tree\nsalmon | fish\nsparrow |",
        "The experiment compared two identical plants, except one received sunlight and the other ",
    )
    qualitative = [
        BasePromptCase(f"qualitative_{index:02d}", "qualitative", prompt, 128, qualitative=True)
        for index, prompt in enumerate(qualitative_prompts)
    ]
    cases = tuple(scored + qualitative)
    _validate_base_prompt_cases(cases)
    return cases


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
    return {"schema": "small-llm-pretraining-base-prompts-v2", "suite_identity": {"prompt_set_id": BASE_PROMPT_SET_ID, "full_scored_cases": 100, "full_qualitative_cases": 20, "full_unique_prompts": 120, "budget": "native per prompt"}, "execution": {"length_bucketed": True, "max_batch_size": BASE_PROMPT_BATCH_SIZE}, "greedy": {"sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "seed": 17}, "summary": _summary(greedy), "cases": greedy}, "sampled": {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 17}, "summary": _summary(sampled), "cases": sampled}}


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


__all__ = ["BASE_PROMPT_CASES_V2", "BASE_PROMPT_SET_ID", "BasePromptCase", "L20_TASKS", "run_base_prompt_suite_v2", "run_l20_conditional_likelihood", "SmallLLMHarnessLM"]