"""Build the ADR-0122 capacity-aware 10% S0 bundle.

This recipe is intentionally specific to the completed 100M/2B parent. It keeps
85% unique instruction targets and 15% ClimbMix replay while consuming the
finite smaller SmolTalk sources once and filling the remaining instruction
budget from Magpie. Validation and test are rebuilt deterministically at the
frozen 4% S0 horizon and must match the historical split identities exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

from dataset.src.remote import sha256_path
from trainer.identity import canonical_hash

from .builder import SFTDatasetBuilder
from .bundle import BUNDLE_SCHEMA_VERSION
from .config import SFTDataConfig
from .sources import JsonlConversationSource, iter_schema_v2_replay
from .template import TiktokenGPT2Encoder

PARENT_CONSUMED_TARGETS = 2_001_000_448
TRAIN_TARGETS = 200_100_044
INSTRUCTION_TARGETS = 170_085_037
REPLAY_TARGETS = 30_015_007
FROZEN_HELDOUT_TARGETS = 2_106_316
SOURCE_TARGET_TOLERANCE = 4_096

PLANNED_TRAIN_SOURCE_TARGETS = {
    "smol-magpie-ultra-short": 160_707_411,
    "smol-contraints": 4_026_530,
    "smollm-rewrite-30k": 3_762_301,
    "smol-summarize-20k": 1_588_795,
    "climbmix-replay": REPLAY_TARGETS,
}

TRAIN_INSTRUCTION_SOURCE_SHARES = {
    source: targets / INSTRUCTION_TARGETS
    for source, targets in PLANNED_TRAIN_SOURCE_TARGETS.items()
    if source != "climbmix-replay"
}

FROZEN_HELDOUT_INSTRUCTION_SOURCE_SHARES = {
    "smol-magpie-ultra-short": 0.75,
    "smol-contraints": 0.10,
    "smollm-rewrite-30k": 0.075,
    "smol-summarize-20k": 0.075,
}

EXPECTED_HELDOUT_MANIFEST_SHA256 = {
    "validation": "26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e",
    "test": "48e99ee51c201da398e227742ca7e023064a408c486cce16e20427d1ec7634d2",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _read_prepared_manifest(prepared: Path) -> dict[str, object]:
    path = prepared / "prepared-manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("prepared SFT source manifest is missing or invalid") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("prepared SFT source manifest must be an object")
    result = dict(payload)
    supplied = result.get("manifest_sha256")
    without_hash = {key: value for key, value in result.items() if key != "manifest_sha256"}
    if supplied != canonical_hash(without_hash):
        raise RuntimeError("prepared SFT source manifest self-hash mismatch")
    raw_files = result.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("prepared SFT source manifest has no file identities")
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise RuntimeError("prepared SFT source file identity is malformed")
        relative = item.get("path")
        digest = item.get("sha256")
        byte_size = item.get("byte_size")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
        ):
            raise RuntimeError("prepared SFT source file identity is malformed")
        file_path = prepared / relative
        if not file_path.is_file() or file_path.is_symlink():
            raise RuntimeError(f"prepared SFT source file is missing or unsafe: {relative}")
        if file_path.stat().st_size != byte_size or sha256_path(file_path) != digest:
            raise RuntimeError(f"prepared SFT source file identity mismatch: {relative}")
    return result


def _augment_report(split_root: Path, report: Mapping[str, object]) -> dict[str, object]:
    without_hash = {key: value for key, value in dict(report).items() if key != "report_sha256"}
    raw_tokens = without_hash.get("actual_source_target_tokens")
    if not isinstance(raw_tokens, Mapping):
        raise RuntimeError("SFT build report has no source target counts")
    source_tokens = {str(source): int(tokens) for source, tokens in raw_tokens.items()}
    total = sum(source_tokens.values())
    if total <= 0:
        raise RuntimeError("SFT build report has no realized target tokens")
    without_hash["actual_source_target_shares"] = {
        source: tokens / total for source, tokens in sorted(source_tokens.items())
    }
    encoded = json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    augmented = {**without_hash, "report_sha256": hashlib.sha256(encoded).hexdigest()}
    (split_root / "build-report.json").write_text(
        json.dumps(augmented, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return augmented


def _verify_train_mix(report: Mapping[str, object]) -> None:
    raw = report.get("actual_source_target_tokens")
    if not isinstance(raw, Mapping):
        raise RuntimeError("10% S0 train report has no source target counts")
    actual = {str(source): int(tokens) for source, tokens in raw.items()}
    if set(actual) != set(PLANNED_TRAIN_SOURCE_TARGETS):
        raise RuntimeError(
            "10% S0 train source set drifted: "
            f"expected={sorted(PLANNED_TRAIN_SOURCE_TARGETS)} actual={sorted(actual)}"
        )
    for source, expected in PLANNED_TRAIN_SOURCE_TARGETS.items():
        observed = actual[source]
        if abs(observed - expected) > SOURCE_TARGET_TOLERANCE:
            raise RuntimeError(
                f"10% S0 source target drift for {source}: expected about {expected}, "
                f"observed {observed}, tolerance {SOURCE_TARGET_TOLERANCE}"
            )
    total = sum(actual.values())
    replay = actual["climbmix-replay"]
    instruction = total - replay
    if abs(total - TRAIN_TARGETS) > SOURCE_TARGET_TOLERANCE:
        raise RuntimeError(
            f"10% S0 realized train horizon drifted: expected {TRAIN_TARGETS}, observed {total}"
        )
    if abs(instruction / total - 0.85) > 5e-5:
        raise RuntimeError(
            "10% S0 top-level instruction/replay mix drifted beyond packing tolerance: "
            f"instruction_share={instruction / total:.8f} replay_share={replay / total:.8f}"
        )


def _build_split(
    *,
    split: str,
    prepared: Path,
    replay_root: Path,
    output: Path,
    optimizer_target_tokens: int,
    context_length: int,
    seed: int,
) -> dict[str, object]:
    is_train = split == "train"
    instruction_source_shares = (
        TRAIN_INSTRUCTION_SOURCE_SHARES
        if is_train
        else FROZEN_HELDOUT_INSTRUCTION_SOURCE_SHARES
    )
    target_loss_tokens = TRAIN_TARGETS if is_train else FROZEN_HELDOUT_TARGETS
    config = SFTDataConfig(
        target_loss_tokens=target_loss_tokens,
        optimizer_target_tokens=optimizer_target_tokens,
        context_length=context_length,
        maximum_assistant_tokens=512,
        instruction_share=0.85 if is_train else 1.0,
        replay_share=0.15 if is_train else 0.0,
        instruction_source_shares=dict(instruction_source_shares),
        seed=seed,
    )
    source_paths = {
        source: prepared / "raw" / split / f"{source}.jsonl"
        for source in instruction_source_shares
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"prepared SFT split is missing source files: {missing}")
    sources = {
        source: JsonlConversationSource(path, expected_source=source, default_source=source)
        for source, path in source_paths.items()
    }
    replay = (
        iter_schema_v2_replay(replay_root, split="train", context_length=context_length)
        if is_train
        else ()
    )
    result = SFTDatasetBuilder(config, encoder=TiktokenGPT2Encoder()).build(
        instruction_sources=sources,
        replay_source=replay,
        output_dir=output,
    )
    report = _augment_report(output, result["report"])
    manifest = result["manifest"]
    if is_train:
        _verify_train_mix(report)
    else:
        expected = EXPECTED_HELDOUT_MANIFEST_SHA256[split]
        observed = manifest["manifest_sha256"]
        if observed != expected:
            raise RuntimeError(
                f"frozen {split} S0 identity drifted: expected {expected}, observed {observed}"
            )
    return {"manifest": manifest, "report": report}


def build_10pct_bundle(
    prepared_dir: Path | str,
    *,
    replay_root: Path | str,
    output_dir: Path | str,
    parent_consumed_tokens: int,
    optimizer_target_tokens: int = 32_768,
    context_length: int = 2_048,
    seed: int = 17,
) -> dict[str, object]:
    if parent_consumed_tokens != PARENT_CONSUMED_TARGETS:
        raise RuntimeError(
            "ADR-0122 10% S0 builder is pinned to parent target count "
            f"{PARENT_CONSUMED_TARGETS}, got {parent_consumed_tokens}"
        )
    prepared = Path(prepared_dir)
    prepared_manifest = _read_prepared_manifest(prepared)
    replay = Path(replay_root)
    replay_manifest_path = replay / "manifest.json"
    if not replay_manifest_path.is_file() or replay_manifest_path.is_symlink():
        raise RuntimeError("replay root has no safe immutable manifest.json")
    replay_manifest_sha256 = sha256_path(replay_manifest_path)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing SFT bundle: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    split_results: dict[str, object] = {}
    try:
        for split in ("train", "validation", "test"):
            split_root = temporary / split
            result = _build_split(
                split=split,
                prepared=prepared,
                replay_root=replay,
                output=split_root,
                optimizer_target_tokens=optimizer_target_tokens,
                context_length=context_length,
                seed=seed,
            )
            split_results[split] = {
                "path": split,
                "manifest_sha256": result["manifest"]["manifest_sha256"],
                "loss_bearing_target_tokens": result["manifest"]["totals"]["loss_bearing_target_tokens"],
                "build_report_sha256": result["report"]["report_sha256"],
            }

        (temporary / "source-manifest.json").write_text(
            json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        recipe = {
            "name": "s0-10pct-capacity-aware-v1",
            "adr": "0122",
            "parent_consumed_targets": PARENT_CONSUMED_TARGETS,
            "train_targets": TRAIN_TARGETS,
            "instruction_targets_planned": INSTRUCTION_TARGETS,
            "replay_targets_planned": REPLAY_TARGETS,
            "planned_train_source_targets": dict(PLANNED_TRAIN_SOURCE_TARGETS),
            "train_instruction_source_shares": dict(TRAIN_INSTRUCTION_SOURCE_SHARES),
            "frozen_heldout_targets": FROZEN_HELDOUT_TARGETS,
            "frozen_heldout_instruction_source_shares": dict(
                FROZEN_HELDOUT_INSTRUCTION_SOURCE_SHARES
            ),
            "expected_heldout_manifest_sha256": dict(EXPECTED_HELDOUT_MANIFEST_SHA256),
            "instruction_repetition": False,
        }
        bundle_without_hash = {
            "schema": "small-llm-sft-bundle",
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "prepared_source_manifest_sha256": prepared_manifest["manifest_sha256"],
            "prepared_source": {
                "dataset_name": prepared_manifest.get("dataset_name"),
                "revision": prepared_manifest.get("revision"),
                "license_id": prepared_manifest.get("license_id"),
                "split_policy": prepared_manifest.get("split_policy"),
            },
            "replay_manifest_sha256": replay_manifest_sha256,
            "train_target_tokens_requested": TRAIN_TARGETS,
            "optimizer_target_tokens": optimizer_target_tokens,
            "context_length": context_length,
            "maximum_assistant_tokens": 512,
            "instruction_share": 0.85,
            "replay_share": 0.15,
            "instruction_source_shares": dict(TRAIN_INSTRUCTION_SOURCE_SHARES),
            "seed": seed,
            "s0_scaling_recipe": recipe,
            "splits": split_results,
        }
        manifest = {**bundle_without_hash, "manifest_sha256": canonical_hash(bundle_without_hash)}
        (temporary / "bundle-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-consumed-tokens", type=_positive_int, required=True)
    parser.add_argument("--optimizer-target-tokens", type=_positive_int, default=32_768)
    parser.add_argument("--context-length", type=_positive_int, default=2_048)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_10pct_bundle(
        args.prepared_dir,
        replay_root=args.replay_root,
        output_dir=args.output_dir,
        parent_consumed_tokens=args.parent_consumed_tokens,
        optimizer_target_tokens=args.optimizer_target_tokens,
        context_length=args.context_length,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "parent_consumed_tokens": args.parent_consumed_tokens,
                "sft_fraction": 0.10,
                "requested_sft_targets": TRAIN_TARGETS,
                "bundle": manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_HELDOUT_MANIFEST_SHA256",
    "FROZEN_HELDOUT_TARGETS",
    "FROZEN_HELDOUT_INSTRUCTION_SOURCE_SHARES",
    "INSTRUCTION_TARGETS",
    "PARENT_CONSUMED_TARGETS",
    "PLANNED_TRAIN_SOURCE_TARGETS",
    "REPLAY_TARGETS",
    "SOURCE_TARGET_TOLERANCE",
    "TRAIN_INSTRUCTION_SOURCE_SHARES",
    "TRAIN_TARGETS",
    "build_10pct_bundle",
]
