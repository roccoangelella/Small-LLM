"""Command-line interface for immutable SFT dataset production and verification."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Mapping, Sequence

from .builder import SFTDatasetBuilder
from .config import SFTDataConfig
from .sources import JsonlConversationSource, iter_schema_v2_replay
from .storage import SFTShardReader
from .template import TiktokenGPT2Encoder


def _load_config(path: Path | None) -> SFTDataConfig:
    if path is None:
        return SFTDataConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid SFT config: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("SFT config must contain a JSON object")
    allowed = {item.name for item in fields(SFTDataConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RuntimeError(f"unknown SFT config fields: {unknown}")
    return SFTDataConfig(**dict(payload))


def _source_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        source, separator, path = value.partition("=")
        if not separator or not source or not path:
            raise ValueError("--source values must use source_name=/path/to/file.jsonl")
        if source in result:
            raise ValueError(f"duplicate --source value for {source!r}")
        result[source] = Path(path)
    return result


def _build(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    source_paths = _source_arguments(args.source)
    configured = set(config.instruction_source_shares)
    if set(source_paths) != configured:
        raise RuntimeError(
            f"--source names must be exactly {sorted(configured)}"
        )
    sources = {
        source: JsonlConversationSource(
            path,
            expected_source=source,
            default_source=source,
        )
        for source, path in source_paths.items()
    }
    replay = iter_schema_v2_replay(
        args.replay_root,
        split="train",
        context_length=config.context_length,
    )
    builder = SFTDatasetBuilder(config, encoder=TiktokenGPT2Encoder())
    result = builder.build(
        instruction_sources=sources,
        replay_source=replay,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    reader = SFTShardReader(
        args.dataset_dir,
        split=args.split,
        verify_checksums=True,
    )
    targets = records = padded_positions = 0
    blocks = 0
    for batch in reader.iter_from_start():
        blocks += 1
        records += batch.sequence_count
        targets += batch.target_token_count
        padded_positions += batch.labels.numel()
    result = {
        "status": "verified",
        "manifest_identity": reader.manifest_identity,
        "split": args.split,
        "blocks": blocks,
        "records": records,
        "loss_bearing_target_tokens": targets,
        "padded_label_positions_examined": padded_positions,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    path = Path(args.dataset_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("manifest must contain an object")
    print(
        json.dumps(
            {
                "manifest_sha256": payload.get("manifest_sha256"),
                "config": payload.get("config"),
                "totals": payload.get("totals"),
                "shards": payload.get("shards"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build immutable S0 data")
    build.add_argument("--config", type=Path)
    build.add_argument(
        "--source",
        action="append",
        default=[],
        help="repeat source_name=/path/to/export.jsonl for every configured source",
    )
    build.add_argument("--replay-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(function=_build)

    verify = subparsers.add_parser("verify", help="verify every SFT shard and block")
    verify.add_argument("--dataset-dir", type=Path, required=True)
    verify.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    verify.set_defaults(function=_verify)

    inspect = subparsers.add_parser("inspect", help="print dataset identity and totals")
    inspect.add_argument("--dataset-dir", type=Path, required=True)
    inspect.set_defaults(function=_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
