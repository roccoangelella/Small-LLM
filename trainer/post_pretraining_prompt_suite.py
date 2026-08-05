"""Print qualitative generations from the best verified remote base checkpoint.

This is deliberately a base-model prompt suite, not an instruction-following
benchmark. Prompts are phrased as document continuations, simple Q/A records,
or short structured examples so the pretrained causal model has a clear text
pattern to continue.
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
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from dataset.src.joint_checkpoint import (
    _verify_published_checkpoint_manifest,
    verify_local_manifest,
)
from dataset.src.remote import HuggingFaceCheckpointStore
from model.config import ModelConfig
from model.model import SmallLLM


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
    suffix = "best" if pointer_name == "best" else "last"
    expected = f"run/{run_id}/checkpoints/{checkpoint_id}/{suffix}"
    if prefix != expected:
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
    _verify_published_checkpoint_manifest(checkpoint_root, embedded_manifest)
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
    """Load verified native weights; pickle is accepted only after hash checks."""

    with (checkpoint_root / "trainer_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
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
) -> list[int]:
    """Generate one continuation with greedy or seeded top-k/top-p sampling."""

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

    device = next(model.parameters()).device
    output = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        with _autocast_context(device, precision):
            logits = model(output[:, -max_seq_len:])[:, -1, :].float()
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
            "Download the best verified Small-LLM base checkpoint and print "
            "qualitative continuations."
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
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if not args.repo_id:
        parser.error("set --repo-id or SMALL_LLM_HF_REPO_ID")
    if args.samples_per_prompt <= 0:
        parser.error("--samples-per-prompt must be positive")
    return args


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
            "the prompt suite requires tiktoken; install the project with .[post-training]"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")

    records: list[dict[str, object]] = []
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
        print("=" * 80)
        print("Small-LLM post-pretraining qualitative prompt suite")
        print(
            json.dumps(
                {
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
                    "sampling": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "seed": args.seed,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )

        for case_index, case in enumerate(cases):
            prompt_ids = encoding.encode(case.prompt, disallowed_special=())
            if len(prompt_ids) > model_config.max_seq_len:
                raise RuntimeError(f"prompt {case.name!r} exceeds the model context window")
            for sample_index in range(args.samples_per_prompt):
                sample_seed = args.seed + case_index * 1_000 + sample_index
                generated_ids = sample_token_ids(
                    model,
                    prompt_ids,
                    max_new_tokens=case.max_new_tokens,
                    max_seq_len=model_config.max_seq_len,
                    eos_token_id=encoding.eot_token,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=sample_seed,
                    precision=precision,
                )
                continuation = encoding.decode(
                    [
                        token_id
                        for token_id in generated_ids
                        if token_id != encoding.eot_token
                    ]
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
                records.append(
                    {
                        "name": case.name,
                        "category": case.category,
                        "sample": sample_index + 1,
                        "seed": sample_seed,
                        "prompt": case.prompt,
                        "continuation": continuation,
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(generated_ids),
                    }
                )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {"checkpoint": checkpoint_info, "results": records},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved machine-readable results to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
