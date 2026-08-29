"""Print qualitative generations or teacher-forced diagnostics from a verified base checkpoint.

This is deliberately a base-model output suite, not an instruction-following
benchmark. Prompts are phrased as document continuations, simple Q/A records,
or short structured examples so the pretrained causal model has a clear text
pattern to continue. A teacher-forced validation mode can instead inspect the
model's raw next-token distribution on held-out validation text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from dataset.src.joint_checkpoint import (
    _verify_published_checkpoint_manifest,
    verify_local_manifest,
)
from dataset.src.remote import HuggingFaceCheckpointStore
from model.config import ModelConfig
from model.model import SmallLLM
from trainer.state import load_trainer_state_file
from trainer.teacher_forced_diagnostic import (
    print_teacher_forced_report,
    run_teacher_forced_validation,
)


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One human-readable qualitative generation case."""

    name: str
    category: str
    prompt: str
    max_new_tokens: int


PROMPT_CASES: tuple[PromptCase, ...] = (
    PromptCase(
        "story_opening",
        "continuation",
        "The rain had stopped before dawn, leaving the streets covered in ",
        128,
    ),
    PromptCase(
        "science_explanation",
        "continuation",
        "Water can exist as a solid, a liquid, or a gas. The transition from liquid water to water vapor occurs when ",
        128,
    ),
    PromptCase(
        "encyclopedia_style",
        "continuation",
        "The Roman Republic was a period of ancient Roman civilization that began after ",
        128,
    ),
    PromptCase(
        "dialogue",
        "continuation",
        "Alice: Did you remember to close the window?\nBen: I thought you had closed it.\nAlice:",
        96,
    ),
    PromptCase(
        "list_pattern",
        "structured",
        "France | Paris\nItaly | Rome\nGermany |",
        64,
    ),
    PromptCase(
        "sentiment_pattern",
        "structured",
        "Text: I loved every minute of the film.\nSentiment: positive\n\nText: The plot was tedious and predictable.\nSentiment: negative\n\nText: The acting was excellent, although the ending was weak.\nSentiment:",
        48,
    ),
    PromptCase("capital_france", "question", "Question: What is the capital of France?\nAnswer:", 48),
    PromptCase("largest_planet", "question", "Question: What is the largest planet in the Solar System?\nAnswer:", 48),
    PromptCase("red_planet", "question", "Question: Which planet is commonly called the Red Planet?\nAnswer:", 48),
    PromptCase("hamlet_author", "question", "Question: Who wrote the play Hamlet?\nAnswer:", 48),
    PromptCase("water_freezing", "question", "Question: At what temperature does pure water freeze on the Celsius scale?\nAnswer:", 48),
    PromptCase("largest_ocean", "question", "Question: What is the largest ocean on Earth?\nAnswer:", 48),
    PromptCase("japan_currency", "question", "Question: What is the currency of Japan?\nAnswer:", 48),
    PromptCase("brazil_language", "question", "Question: What is the main language spoken in Brazil?\nAnswer:", 48),
    PromptCase("blood_pump", "question", "Question: Which organ pumps blood through the human body?\nAnswer:", 48),
    PromptCase("photosynthesis", "question", "Question: What process do plants use to convert light energy into chemical energy?\nAnswer:", 64),
    PromptCase("leap_year", "question", "Question: How many days are in a leap year?\nAnswer:", 48),
    PromptCase("simple_arithmetic", "question", "Question: What is 7 multiplied by 8?\nAnswer:", 48),
)

_RUN_POINTER_RE = re.compile(r"^run/([^/]+)/(best|latest)\.json$")


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _discover_run_id(store: HuggingFaceCheckpointStore, *, pointer: str) -> str:
    list_files = getattr(store.api, "list_repo_files", None)
    if not callable(list_files):
        raise RuntimeError("--run-id is required because this Hub API cannot list repository files")
    matches: set[str] = set()
    for name in list_files(**store._hub_kwargs()):
        if not isinstance(name, str):
            continue
        match = _RUN_POINTER_RE.fullmatch(name)
        if match is not None and match.group(2) == pointer:
            matches.add(match.group(1))
    if not matches:
        raise RuntimeError(f"the repository contains no run/*/{pointer}.json pointer")
    if len(matches) != 1:
        raise RuntimeError(
            f"the repository contains multiple {pointer} pointers; pass --run-id explicitly: {sorted(matches)}"
        )
    return next(iter(matches))


def _checkpoint_prefix(
    pointer: Mapping[str, object],
    *,
    run_id: str,
    pointer_name: str,
) -> tuple[str, str]:
    checkpoint_id = pointer.get("checkpoint_id")
    if (
        not isinstance(checkpoint_id, str)
        or not checkpoint_id
        or "/" in checkpoint_id
        or "\\" in checkpoint_id
        or checkpoint_id in {".", ".."}
    ):
        raise RuntimeError(f"run/{run_id}/{pointer_name}.json has an invalid checkpoint_id")
    key = "best_prefix" if pointer_name == "best" else "last_prefix"
    prefix = pointer.get(key)
    if not isinstance(prefix, str):
        raise RuntimeError(f"run/{run_id}/{pointer_name}.json has no valid {key}")
    expected_last = f"run/{run_id}/checkpoints/{checkpoint_id}/last"
    expected = (
        {expected_last, f"run/{run_id}/checkpoints/{checkpoint_id}/best"}
        if pointer_name == "best"
        else {expected_last}
    )
    if prefix not in expected:
        raise RuntimeError(f"run/{run_id}/{pointer_name}.json prefix does not match its checkpoint ID")
    return checkpoint_id, prefix


def download_verified_checkpoint(
    *,
    repo_id: str,
    run_id: str | None,
    token: str | None,
    revision: str | None,
    pointer_name: str,
    destination: Path,
) -> tuple[Path, dict[str, object]]:
    """Download and verify one best/latest custom joint checkpoint tree."""

    store = HuggingFaceCheckpointStore(
        repo_id,
        token=token,
        private=True,
        revision=revision,
    )
    selected_run_id = run_id or _discover_run_id(store, pointer=pointer_name)
    pointer_path = f"run/{selected_run_id}/{pointer_name}.json"
    pointer = store.read_json(pointer_path)
    if pointer is None:
        raise RuntimeError(f"Hugging Face pointer is missing: {pointer_path}")
    checkpoint_id, prefix = _checkpoint_prefix(
        pointer,
        run_id=selected_run_id,
        pointer_name=pointer_name,
    )
    checkpoint_root = destination / checkpoint_id
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    store.download_tree(prefix, checkpoint_root)
    verify_local_manifest(checkpoint_root)
    embedded_manifest = _json_object(
        checkpoint_root / "checkpoint_manifest.json",
        label="checkpoint_manifest.json",
    )
    pointer_manifest = pointer.get("checkpoint_manifest")
    supplied_manifest = (
        pointer_manifest if isinstance(pointer_manifest, Mapping) else embedded_manifest
    )
    _verify_published_checkpoint_manifest(checkpoint_root, supplied_manifest)
    return checkpoint_root, {
        "repo_id": repo_id,
        "run_id": selected_run_id,
        "pointer": pointer_name,
        "checkpoint_id": checkpoint_id,
        "prefix": prefix,
        "metric": pointer.get("metric"),
    }


def _normalize_model_config(raw: Mapping[str, object]) -> ModelConfig:
    values = dict(raw)
    pattern = values.get("layer_pattern")
    if isinstance(pattern, list):
        values["layer_pattern"] = tuple(pattern)
    return ModelConfig(**values)  # type: ignore[arg-type]


def _load_model(
    checkpoint_root: Path,
    *,
    device: torch.device,
    model_config_json: Path | None,
) -> tuple[SmallLLM, ModelConfig, Mapping[str, object]]:
    """Load verified native weights from historical pickle or streamed torch checkpoints."""

    state = load_trainer_state_file(
        checkpoint_root / "trainer_state.pkl",
        map_location="cpu",
    )
    if not isinstance(state, Mapping) or state.get("version") != 1:
        raise RuntimeError("trainer_state.pkl has an unsupported structure or version")
    model_state = state.get("model")
    if not isinstance(model_state, Mapping):
        raise RuntimeError("trainer_state.pkl has no model state mapping")

    if model_config_json is not None:
        raw_config = _json_object(model_config_json, label="model config override")
    else:
        raw = state.get("model_config")
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "checkpoint has no self-describing model_config; pass --model-config-json "
                "for checkpoints created before this prompt suite was added"
            )
        raw_config = dict(raw)
    config = _normalize_model_config(raw_config)
    model = SmallLLM(config)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model, config, state


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError(f"{precision} generation requires a CUDA device")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _filter_logits(logits: Tensor, *, top_k: int, top_p: float) -> Tensor:
    filtered = logits
    if top_k > 0:
        k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            -1,
            sorted_indices,
            sorted_logits,
        )
    return filtered


def _generation_budget(case: PromptCase, max_new_tokens: int | None) -> int:
    """Return the prompt's native budget capped by an optional global limit."""

    if max_new_tokens is None:
        return case.max_new_tokens
    return min(case.max_new_tokens, max_new_tokens)


def _decode_token(encoding: Any, token_id: int) -> str:
    return encoding.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")


def _decode_token_trace(
    trace: Sequence[Mapping[str, object]],
    encoding: Any,
) -> list[dict[str, object]]:
    decoded: list[dict[str, object]] = []
    for entry in trace:
        candidates = entry.get("top_tokens")
        if not isinstance(candidates, list):
            raise RuntimeError("token trace has an invalid top_tokens payload")
        decoded_candidates: list[dict[str, object]] = []
        for rank, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise RuntimeError("token trace candidate is not a mapping")
            token_id = int(candidate["token_id"])
            decoded_candidates.append(
                {
                    "rank": rank,
                    "token_id": token_id,
                    "token": _decode_token(encoding, token_id),
                    "probability": float(candidate["probability"]),
                }
            )
        chosen_token_id = int(entry["chosen_token_id"])
        decoded.append(
            {
                "step": int(entry["step"]),
                "chosen_token_id": chosen_token_id,
                "chosen_token": _decode_token(encoding, chosen_token_id),
                "chosen_probability": float(entry["chosen_probability"]),
                "top_tokens": decoded_candidates,
            }
        )
    return decoded


@torch.inference_mode()
def sample_token_ids(
    model: nn.Module,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    precision: str,
    trace_top_tokens: int = 0,
    trace_out: list[dict[str, object]] | None = None,
) -> list[int]:
    """Generate one continuation and optionally capture the raw next-token distribution."""

    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens < 0 or max_seq_len <= 0:
        raise ValueError("generation lengths are invalid")
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if trace_top_tokens < 0:
        raise ValueError("trace_top_tokens must be non-negative")
    if trace_top_tokens > 0 and trace_out is None:
        raise ValueError("trace_out is required when trace_top_tokens is positive")

    device = next(model.parameters()).device
    output = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated: list[int] = []
    for step_index in range(max_new_tokens):
        with _autocast_context(device, precision):
            logits = model(output[:, -max_seq_len:])[:, -1, :].float()

        raw_probabilities: Tensor | None = None
        trace_ids: Tensor | None = None
        trace_probabilities: Tensor | None = None
        if trace_top_tokens > 0:
            raw_probabilities = torch.softmax(logits, dim=-1)
            trace_count = min(trace_top_tokens, raw_probabilities.shape[-1])
            trace_probabilities, trace_ids = torch.topk(
                raw_probabilities,
                trace_count,
                dim=-1,
            )

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            filtered = _filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
            probabilities = torch.softmax(filtered, dim=-1)
            if not torch.isfinite(probabilities).all() or bool(
                (probabilities.sum(dim=-1) <= 0).any()
            ):
                raise FloatingPointError("sampling produced an invalid probability distribution")
            next_token = torch.multinomial(probabilities, 1, generator=generator)

        token_id = int(next_token.item())
        if trace_top_tokens > 0:
            assert trace_out is not None
            assert raw_probabilities is not None
            assert trace_ids is not None
            assert trace_probabilities is not None
            trace_out.append(
                {
                    "step": step_index + 1,
                    "chosen_token_id": token_id,
                    "chosen_probability": float(raw_probabilities[0, token_id].item()),
                    "top_tokens": [
                        {
                            "token_id": int(candidate_id),
                            "probability": float(candidate_probability),
                        }
                        for candidate_id, candidate_probability in zip(
                            trace_ids[0].tolist(),
                            trace_probabilities[0].tolist(),
                            strict=True,
                        )
                    ],
                }
            )

        generated.append(token_id)
        output = torch.cat((output, next_token), dim=1)
        if token_id == eos_token_id:
            break
    return generated


def _selected_cases(
    *,
    questions_only: bool,
    max_cases: int | None,
) -> tuple[PromptCase, ...]:
    cases = tuple(
        case for case in PROMPT_CASES if not questions_only or case.category == "question"
    )
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("--max-cases must be positive")
        cases = cases[:max_cases]
    return cases


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
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a verified Small-LLM base checkpoint and run qualitative "
            "generation or held-out teacher-forced output diagnostics."
        )
    )
    parser.add_argument("--repo-id", default=os.environ.get("SMALL_LLM_HF_REPO_ID"))
    parser.add_argument("--run-id", default=os.environ.get("SMALL_LLM_RUN_ID"))
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--revision")
    parser.add_argument("--pointer", choices=("best", "latest"), default="best")
    parser.add_argument("--model-config-json", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--questions-only", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="cap every prompt's native generation budget at this many new tokens",
    )
    parser.add_argument(
        "--trace-top-tokens",
        type=int,
        default=0,
        help="print and save the top-N raw next-token probabilities at every generation step",
    )
    parser.add_argument(
        "--teacher-forced-validation",
        nargs="?",
        const="auto",
        metavar="DATASET_DIR",
        help=(
            "run the held-out teacher-forced confidence diagnostic instead of prompt "
            "generation; optionally provide the dataset root, otherwise auto-match "
            "the attached Kaggle dataset to the checkpoint"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if not args.repo_id:
        parser.error("set --repo-id or SMALL_LLM_HF_REPO_ID")
    if args.samples_per_prompt <= 0:
        parser.error("--samples-per-prompt must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.trace_top_tokens < 0:
        parser.error("--trace-top-tokens must be non-negative")
    return args


def _checkpoint_display(
    checkpoint_info: Mapping[str, object],
    *,
    device: torch.device,
    precision: str,
    trainer_state: Mapping[str, object],
    model_config: ModelConfig,
) -> dict[str, object]:
    return {
        **checkpoint_info,
        "device": str(device),
        "precision": precision,
        "global_step": trainer_state.get("global_step"),
        "consumed_tokens": trainer_state.get("consumed_tokens"),
        "model_config": {
            "d_model": model_config.d_model,
            "n_layers": model_config.n_layers,
            "d_ff": model_config.d_ff,
            "architecture": model_config.architecture,
            "max_seq_len": model_config.max_seq_len,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get(args.token_env)
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    cases = _selected_cases(
        questions_only=args.questions_only,
        max_cases=args.max_cases,
    )

    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "the output suite requires tiktoken; install the project with .[post-training]"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")

    sampling_info = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "trace_top_tokens": args.trace_top_tokens,
    }
    records: list[dict[str, object]] = []
    output_payload: dict[str, object] | None = None
    with tempfile.TemporaryDirectory(prefix="small-llm-prompt-suite-") as temporary:
        checkpoint_root, checkpoint_info = download_verified_checkpoint(
            repo_id=str(args.repo_id),
            run_id=args.run_id,
            token=token,
            revision=args.revision,
            pointer_name=args.pointer,
            destination=Path(temporary),
        )
        model, model_config, trainer_state = _load_model(
            checkpoint_root,
            device=device,
            model_config_json=args.model_config_json,
        )
        checkpoint_display = _checkpoint_display(
            checkpoint_info,
            device=device,
            precision=precision,
            trainer_state=trainer_state,
            model_config=model_config,
        )

        if args.teacher_forced_validation is not None:
            print("=" * 80)
            print("Small-LLM post-pretraining model-output suite")
            print(json.dumps(checkpoint_display, indent=2, sort_keys=True))
            teacher_forced_report = run_teacher_forced_validation(
                model,
                model_config=model_config,
                checkpoint_root=checkpoint_root,
                dataset_request=str(args.teacher_forced_validation),
                device=device,
                precision=precision,
                encoding=encoding,
            )
            print_teacher_forced_report(teacher_forced_report)
            output_payload = {
                "checkpoint": checkpoint_info,
                "teacher_forced_validation": teacher_forced_report,
            }
        else:
            print("=" * 80)
            print("Small-LLM post-pretraining qualitative prompt suite")
            print(
                json.dumps(
                    {
                        **checkpoint_display,
                        "sampling": sampling_info,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

            for case_index, case in enumerate(cases):
                prompt_ids = encoding.encode(case.prompt, disallowed_special=())
                if len(prompt_ids) > model_config.max_seq_len:
                    raise RuntimeError(f"prompt {case.name!r} exceeds the model context window")
                generation_budget = _generation_budget(case, args.max_new_tokens)
                for sample_index in range(args.samples_per_prompt):
                    sample_seed = args.seed + case_index * 1_000 + sample_index
                    raw_trace: list[dict[str, object]] = []
                    generated_ids = sample_token_ids(
                        model,
                        prompt_ids,
                        max_new_tokens=generation_budget,
                        max_seq_len=model_config.max_seq_len,
                        eos_token_id=encoding.eot_token,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=sample_seed,
                        precision=precision,
                        trace_top_tokens=args.trace_top_tokens,
                        trace_out=raw_trace if args.trace_top_tokens > 0 else None,
                    )
                    continuation = encoding.decode(
                        [
                            token_id
                            for token_id in generated_ids
                            if token_id != encoding.eot_token
                        ]
                    )
                    token_trace = (
                        _decode_token_trace(raw_trace, encoding)
                        if args.trace_top_tokens > 0
                        else []
                    )
                    print("\n" + "-" * 80)
                    print(
                        f"[{case.category}] {case.name} | sample={sample_index + 1} "
                        f"| seed={sample_seed}"
                    )
                    print("PROMPT:")
                    print(case.prompt)
                    print("\nCONTINUATION:")
                    print(continuation)
                    if token_trace:
                        print("\nTOKEN TRACE (raw model probabilities before decoding filters):")
                        for trace_entry in token_trace:
                            print(
                                f"step {int(trace_entry['step']):02d} "
                                f"chosen={trace_entry['chosen_token']!r} "
                                f"p={float(trace_entry['chosen_probability']):.6f}"
                            )
                            candidates = trace_entry["top_tokens"]
                            assert isinstance(candidates, list)
                            print(
                                "  top: "
                                + " | ".join(
                                    f"{candidate['token']!r}={float(candidate['probability']):.6f}"
                                    for candidate in candidates
                                    if isinstance(candidate, Mapping)
                                )
                            )
                    record: dict[str, object] = {
                        "name": case.name,
                        "category": case.category,
                        "sample": sample_index + 1,
                        "seed": sample_seed,
                        "prompt": case.prompt,
                        "continuation": continuation,
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(generated_ids),
                        "max_new_tokens": generation_budget,
                    }
                    if token_trace:
                        record["token_trace"] = token_trace
                    records.append(record)

            output_payload = {
                "checkpoint": checkpoint_info,
                "sampling": sampling_info,
                "results": records,
            }

    if args.output_json is not None:
        assert output_payload is not None
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved machine-readable results to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
