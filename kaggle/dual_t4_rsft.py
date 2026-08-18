#!/usr/bin/env python3
"""Adapt the qualified 2xT4 SFT runtime to R-SFT without forking its DDP engine.

The base ``dual_t4_sft.py`` remains responsible for the qualified Kaggle T4
runtime, exact global-token optimizer step, FP16 prewarm, DDP synchronization,
checkpoint cadence, resume, W&B, and remote publication.  This adapter changes
only R-SFT semantics immediately before the shared trainer is entered:

* the completed S0 checkpoint is accepted as the parent;
* the three padded rows are promoted to semantic reasoning-token rows;
* the immutable bundle's requested target count is the exact one-pass budget;
* R-SFT tokenizer/delimiter identity is carried in checkpoints;
* the S0 behavior probe is marked skipped until the R-SFT suite is frozen.
"""
from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

RSFT_STAGE = "r_sft_r0"
DELIMITER_FORMATS = ("atomic", "textual")


def _arguments(argv: Sequence[str] | None) -> tuple[Path, str, Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--rsft-delimiter-format", choices=DELIMITER_FORMATS, required=True)
    parser.add_argument("--rsft-token-spec", type=Path, required=True)
    args, trainer_argv = parser.parse_known_args(argv)
    worktree = args.worktree.resolve()
    if not (worktree / "trainer").is_dir() or not (worktree / "post_training" / "R-SFT").is_dir():
        raise SystemExit(f"pinned R-SFT worktree is invalid: {worktree}")
    token_spec = args.rsft_token_spec.expanduser().resolve()
    if not token_spec.is_file() or token_spec.is_symlink():
        raise SystemExit(f"R-SFT token spec is missing or unsafe: {token_spec}")
    return worktree, str(args.rsft_delimiter_format), token_spec, list(trainer_argv)


def _argument_value(argv: Sequence[str], flag: str, *, default: str | None = None) -> str:
    values = list(argv)
    try:
        index = values.index(flag)
    except ValueError:
        if default is not None:
            return default
        raise RuntimeError(f"required trainer argument is missing: {flag}") from None
    if index + 1 >= len(values):
        raise RuntimeError(f"trainer argument has no value: {flag}")
    return str(values[index + 1])


def bundle_target_budget(dataset_dir: Path | str) -> int:
    """Read the exact one-pass loss-bearing target budget from an immutable bundle."""

    path = Path(dataset_dir) / "bundle-manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"R-SFT bundle manifest is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("R-SFT bundle manifest must be a JSON object")
    value = payload.get("train_target_tokens_requested")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("R-SFT bundle has no positive train_target_tokens_requested")
    return value


def _load_rsft_module(worktree: Path, name: str) -> ModuleType:
    module_name = f"small_llm_kaggle_rsft_{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = worktree / "post_training" / "R-SFT" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_reasoning_token_spec(path: Path | str, tokenizer_module: ModuleType) -> Any:
    """Load either the compact three-string file or full tokenizer metadata."""

    token_path = Path(path)
    try:
        payload = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"R-SFT reasoning token spec is missing or invalid: {token_path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("R-SFT reasoning token spec must be a JSON object")

    nested = payload.get(tokenizer_module.TOKENIZER_METADATA_KEY)
    if isinstance(nested, Mapping):
        payload = nested

    compact_keys = {"reasoning_start", "reasoning_end", "answer_start"}
    if set(payload) == compact_keys:
        try:
            return tokenizer_module.ReasoningTokenSpec(
                reasoning_start=payload["reasoning_start"],
                reasoning_end=payload["reasoning_end"],
                answer_start=payload["answer_start"],
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"invalid compact R-SFT token spec: {error}") from error
    try:
        return tokenizer_module.ReasoningTokenSpec.from_metadata(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid R-SFT tokenizer metadata: {error}") from error


def rsft_pipeline_identity(
    *,
    parent_identity: Mapping[str, object],
    bundle_manifest: Mapping[str, object],
    delimiter_format: str,
    token_metadata: Mapping[str, object],
) -> dict[str, object]:
    if delimiter_format not in DELIMITER_FORMATS:
        raise ValueError(f"unsupported R-SFT delimiter format: {delimiter_format}")
    return {
        "stage": RSFT_STAGE,
        "parent_checkpoint_identity": parent_identity["identity_sha256"],
        "bundle_manifest_identity": bundle_manifest["manifest_sha256"],
        "template_identity": f"small-llm-rsft-r0-{delimiter_format}-v1",
        "loss_identity": "assistant-only-ce-v1",
        "delimiter_format": delimiter_format,
        "reasoning_tokenizer": dict(token_metadata),
    }


def _skipped_behavior() -> dict[str, object]:
    return {
        "schema": "small-llm-rsft-inline-behavior-skipped-v1",
        "cases": [],
        "summary": {
            "skipped": True,
            "reason": "R-SFT qualification suite is intentionally deferred",
            "pass_rate": None,
            "eos_termination_rate": None,
            "runaway_rate": None,
        },
    }


def _rewrite_training_summary(
    checkpoint_dir: Path,
    *,
    exact_targets: int,
    delimiter_format: str,
    token_metadata: Mapping[str, object],
) -> dict[str, object]:
    source = checkpoint_dir / "sft-summary.json"
    if not source.is_file():
        raise RuntimeError(f"shared SFT trainer did not produce its summary: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("shared SFT trainer summary is malformed")
    payload["schema"] = "small-llm-rsft-training-summary-v1"
    run_id = payload.pop("sft_run_id", None)
    payload["rsft_run_id"] = run_id
    payload["stage"] = RSFT_STAGE
    payload["delimiter_format"] = delimiter_format
    payload["reasoning_tokenizer"] = dict(token_metadata)
    budget = payload.get("budget")
    if not isinstance(budget, dict):
        budget = {}
        payload["budget"] = budget
    budget["mode"] = "bundle-exact-one-pass"
    budget["fraction"] = None
    budget["requested_loss_bearing_target_tokens"] = exact_targets
    payload["behavior"] = _skipped_behavior()["summary"]
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    source.write_text(rendered, encoding="utf-8")
    (checkpoint_dir / "r-sft-summary.json").write_text(rendered, encoding="utf-8")
    return payload


def _run_shared_trainer(
    original_main: Any,
    trainer_argv: Sequence[str],
    *,
    worktree: Path,
    delimiter_format: str,
    token_spec_path: Path,
) -> int:
    import post_training.sft.checkpoints as sft_checkpoints
    import post_training.sft.train_cli as sft_train

    tokenizer = _load_rsft_module(worktree, "tokenizer")
    transition = _load_rsft_module(worktree, "model_transition")
    token_spec = load_reasoning_token_spec(token_spec_path, tokenizer)
    token_metadata = token_spec.to_metadata()

    dataset_dir = Path(_argument_value(trainer_argv, "--dataset-dir")).resolve()
    checkpoint_dir = Path(_argument_value(trainer_argv, "--checkpoint-dir")).resolve()
    seed = int(_argument_value(trainer_argv, "--seed", default="17"))
    exact_targets = bundle_target_budget(dataset_dir)

    base_reader = sft_train.SFTShardReader
    current_print = sft_train.print

    class RSFTShardReader(base_reader):
        def pipeline_state(self) -> dict[str, object]:
            state = dict(super().pipeline_state())
            state[tokenizer.TOKENIZER_METADATA_KEY] = token_metadata
            state["rsft_format"] = {
                "version": 1,
                "stage": RSFT_STAGE,
                "delimiter_format": delimiter_format,
            }
            return state

    def exact_budget(_: int, **__: Any) -> int:
        return exact_targets

    def promoted_loader(root: Path | str, *, device: Any = "cpu"):
        parent_model, parent_config, parent_identity = sft_checkpoints.load_verified_native_checkpoint(
            root,
            device=device,
        )
        promoted_model, promoted_config = transition.promote_s0_model_for_rsft(
            parent_model,
            parent_config,
            seed=seed,
        )
        return promoted_model, promoted_config, parent_identity

    def checkpoint_hashes(**kwargs: Any):
        return sft_checkpoints.sft_checkpoint_hashes(
            **kwargs,
            template_identity=f"small-llm-rsft-r0-{delimiter_format}-v1",
        )

    def pipeline_identity(*, parent_identity: Mapping[str, object], bundle_manifest: Mapping[str, object]):
        return rsft_pipeline_identity(
            parent_identity=parent_identity,
            bundle_manifest=bundle_manifest,
            delimiter_format=delimiter_format,
            token_metadata=token_metadata,
        )

    def rsft_behavior(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args, kwargs
        return _skipped_behavior()

    def rsft_print(*args: Any, **kwargs: Any) -> None:
        # Suppress the shared trainer's fraction-shaped final summary. The
        # corrected bundle-exact R-SFT summary is emitted below on rank zero.
        if len(args) == 1 and isinstance(args[0], str):
            try:
                payload = json.loads(args[0])
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict) and "sft_summary" in payload:
                return
        current_print(*args, **kwargs)

    sft_train.sft_budget_from_parent = exact_budget
    sft_train.download_parent_checkpoint = sft_checkpoints.download_parent_checkpoint
    sft_train.load_verified_native_checkpoint = promoted_loader
    sft_train.sft_checkpoint_hashes = checkpoint_hashes
    sft_train._checkpoint_pipeline_identity = pipeline_identity
    sft_train.SFTShardReader = RSFTShardReader
    sft_train.evaluate_behavior = rsft_behavior
    sft_train.print = rsft_print

    exit_code = int(original_main(list(trainer_argv)))
    if int(os.environ.get("RANK", "0")) == 0 and exit_code == 0:
        summary = _rewrite_training_summary(
            checkpoint_dir,
            exact_targets=exact_targets,
            delimiter_format=delimiter_format,
            token_metadata=token_metadata,
        )
        current_print(json.dumps({"rsft_summary": summary}, sort_keys=True), flush=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    worktree, delimiter_format, token_spec_path, trainer_argv = _arguments(argv)
    sys.path.insert(0, str(worktree))

    import dual_t4_sft as shared
    import post_training.sft.train_cli as sft_train

    original_train_main = sft_train.main
    original_summary_rewrite = shared._rewrite_summary_fraction
    original_shared_print = getattr(shared, "print", builtins.print)

    def shared_print(*args: Any, **kwargs: Any) -> None:
        if args and isinstance(args[0], str) and args[0].startswith("[kaggle-sft-ddp] execution:"):
            builtins.print(
                "[kaggle-rsft-ddp] execution: 2x Tesla T4, shared exact-token SFT engine, "
                f"delimiter_format={delimiter_format}, bundle-exact one-pass budget",
                flush=True,
            )
            return
        original_shared_print(*args, **kwargs)

    def proxy_main(inner_argv: Sequence[str] | None = None) -> int:
        return _run_shared_trainer(
            original_train_main,
            list(inner_argv or ()),
            worktree=worktree,
            delimiter_format=delimiter_format,
            token_spec_path=token_spec_path,
        )

    # The underlying DDP wrapper requires a syntactically valid fraction even
    # though R-SFT replaces the budget at trainer entry. Keep it private and
    # prevent its post-run fraction rewrite from touching the R-SFT summary.
    sft_train.main = proxy_main
    shared._rewrite_summary_fraction = lambda *args, **kwargs: None
    shared.print = shared_print
    shared_argv = [
        "--worktree",
        str(worktree),
        "--sft-fraction-numerator",
        "1",
        "--sft-fraction-denominator",
        "2",
        *trainer_argv,
    ]
    try:
        return int(shared.main(shared_argv))
    finally:
        sft_train.main = original_train_main
        shared._rewrite_summary_fraction = original_summary_rewrite
        shared.print = original_shared_print


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DELIMITER_FORMATS",
    "RSFT_STAGE",
    "bundle_target_budget",
    "load_reasoning_token_spec",
    "main",
    "rsft_pipeline_identity",
]
