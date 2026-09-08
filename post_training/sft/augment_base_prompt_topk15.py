"""Augment an existing full SFT qualification with Base Prompt top-k15 samples only.

This avoids rerunning eval_core or the large SFT Behavior suites when comparing
canonical full-distribution sampling (top_k=0) against the tighter top_k=15
contract from ADR 0153. The command reloads the exact parent and SFT checkpoints,
verifies their identities against the source qualification, generates only the
new Base Prompt view, and writes a new qualification JSON that remains directly
consumable by ``trainer.base_prompt_judge``.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from trainer.identity import canonical_hash
from trainer.pretraining_eval_v2 import (
    BASE_PROMPT_SET_ID,
    BASE_PROMPT_TOPK15,
    _run_prompt_view,
    _summary,
)

from .checkpoints import load_verified_native_checkpoint
from .eval_suite import _resolve, _resolve_device, _resolve_precision


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--overwrite-view", action="store_true")

    parent = parser.add_argument_group("parent checkpoint")
    parent.add_argument("--parent-checkpoint-dir", type=Path)
    parent.add_argument("--parent-repo-id")
    parent.add_argument("--parent-run-id")
    parent.add_argument("--parent-pointer", choices=("best", "latest"), default="best")
    parent.add_argument("--parent-revision")

    tuned = parser.add_argument_group("SFT checkpoint")
    tuned.add_argument("--sft-checkpoint-dir", type=Path)
    tuned.add_argument("--sft-repo-id")
    tuned.add_argument("--sft-run-id")
    tuned.add_argument("--sft-pointer", choices=("best", "latest"), default="latest")
    tuned.add_argument("--sft-revision")
    return parser.parse_args(argv)


def _load_source(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read qualification JSON {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("qualification JSON root must be an object")
    result = dict(payload)
    if result.get("schema") != "small-llm-post-sft-qualification-v2":
        raise RuntimeError("top-k15 augmenter requires a post-SFT qualification-v2 JSON")
    if result.get("suite") != "full":
        raise RuntimeError("top-k15 comparison requires the full 120-case Base Prompt suite")
    return result


def _source_side(report: Mapping[str, object], label: str) -> Mapping[str, object]:
    side = report.get(label)
    if not isinstance(side, Mapping):
        raise RuntimeError(f"source qualification has no {label!r} side")
    scorecard = side.get("scorecard")
    if not isinstance(scorecard, Mapping):
        raise RuntimeError(f"source qualification {label!r} side has no scorecard")
    suite = scorecard.get("base_prompt_suite_v2")
    if not isinstance(suite, Mapping):
        raise RuntimeError(f"source qualification {label!r} side has no Base Prompt v2")
    identity = suite.get("suite_identity")
    if not isinstance(identity, Mapping) or identity.get("prompt_set_id") != BASE_PROMPT_SET_ID:
        raise RuntimeError(f"source qualification {label!r} uses the wrong Base Prompt set")
    return side


def _verify_checkpoint_identity(
    source_side: Mapping[str, object],
    loaded_identity: Mapping[str, object],
    *,
    label: str,
) -> None:
    expected = source_side.get("checkpoint")
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"source qualification {label!r} side has no checkpoint identity")
    for key in ("checkpoint_id", "identity_sha256"):
        expected_value = expected.get(key)
        loaded_value = loaded_identity.get(key)
        if (
            isinstance(expected_value, str)
            and isinstance(loaded_value, str)
            and expected_value != loaded_value
        ):
            raise RuntimeError(
                f"{label} checkpoint mismatch for {key}: "
                f"source={expected_value!r}, loaded={loaded_value!r}"
            )


def _generate_topk15(model, *, max_seq_len: int, precision: str) -> dict[str, object]:
    rows = _run_prompt_view(
        model,
        model_max_seq_len=max_seq_len,
        precision=precision,
        suite="full",
        temperature=1.0,
        top_p=1.0,
        top_k=BASE_PROMPT_TOPK15,
        seed=17,
    )
    return {
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": BASE_PROMPT_TOPK15,
            "seed": 17,
        },
        "summary": _summary(rows),
        "cases": rows,
    }


def _attach_view(
    report: dict[str, object],
    *,
    label: str,
    view: Mapping[str, object],
    overwrite: bool,
) -> None:
    side = report[label]
    assert isinstance(side, dict)
    scorecard = side["scorecard"]
    assert isinstance(scorecard, dict)
    suite = scorecard["base_prompt_suite_v2"]
    assert isinstance(suite, dict)
    if "sampled_topk15" in suite and not overwrite:
        raise RuntimeError(
            f"{label} Base Prompt suite already contains sampled_topk15; "
            "pass --overwrite-view to replace it"
        )
    suite["sampled_topk15"] = dict(view)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    source_path = args.input.resolve()
    report = deepcopy(_load_source(source_path))
    parent_source = _source_side(report, "parent")
    sft_source = _source_side(report, "sft")

    token = os.environ.get(args.token_env)
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 Base Prompt generation requires CUDA")

    parent_root, _ = _resolve(
        local=args.parent_checkpoint_dir,
        repo_id=args.parent_repo_id,
        run_id=args.parent_run_id,
        pointer=args.parent_pointer,
        revision=args.parent_revision,
        token=token,
        label="parent-topk15",
    )
    sft_root, _ = _resolve(
        local=args.sft_checkpoint_dir,
        repo_id=args.sft_repo_id,
        run_id=args.sft_run_id,
        pointer=args.sft_pointer,
        revision=args.sft_revision,
        token=token,
        label="sft-topk15",
    )

    print("Generating parent Base Prompt sampled_topk15 view...", flush=True)
    parent_model, parent_config, parent_identity = load_verified_native_checkpoint(
        parent_root, device=device
    )
    _verify_checkpoint_identity(parent_source, parent_identity, label="parent")
    parent_view = _generate_topk15(
        parent_model,
        max_seq_len=parent_config.max_seq_len,
        precision=precision,
    )
    del parent_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Generating SFT Base Prompt sampled_topk15 view...", flush=True)
    sft_model, sft_config, sft_identity = load_verified_native_checkpoint(
        sft_root, device=device
    )
    _verify_checkpoint_identity(sft_source, sft_identity, label="sft")
    if parent_config != sft_config:
        raise RuntimeError("parent and SFT checkpoints have different model geometry")
    sft_view = _generate_topk15(
        sft_model,
        max_seq_len=sft_config.max_seq_len,
        precision=precision,
    )
    del sft_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _attach_view(
        report,
        label="parent",
        view=parent_view,
        overwrite=args.overwrite_view,
    )
    _attach_view(
        report,
        label="sft",
        view=sft_view,
        overwrite=args.overwrite_view,
    )
    report["base_prompt_topk15_augmentation"] = {
        "source_file": source_path.name,
        "contract": {
            "view": "sampled_topk15",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": BASE_PROMPT_TOPK15,
            "seed": 17,
        },
        "checkpoint_identity_verified": True,
        "reran_other_evaluation_layers": False,
    }
    report_without_hash = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    report["report_sha256"] = canonical_hash(report_without_hash)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved top-k15 augmented qualification to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
