"""Regression tests for the unified finite-dataset profile surface."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.qualification import (
    PROFILES,
    derive_plan,
    get_profile,
    main,
    production_arguments,
)


def _shard(split: str, first: int, last: int, sequences: int) -> dict[str, object]:
    tokens = sequences * 2049
    return {
        "filename": f"{split}/{split}-{first:06d}.bin",
        "split": split,
        "byte_size": tokens * 2,
        "token_count": tokens,
        "sequence_count": sequences,
        "checksum": "a" * 64,
        "first_block_id": first,
        "last_block_id": last,
    }


def _manifest(
    profile_key: str,
    train_blocks: int,
    validation_blocks: int = 4,
    *,
    train_sequences: int | None = None,
) -> dict[str, object]:
    profile = get_profile(profile_key)
    accepted = profile.target_source_tokens + 100
    validation_source = max(1, profile.target_source_tokens // 1_000)
    if train_sequences is None:
        train_sequences = train_blocks * profile.sequences_per_block
    return {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": profile.context_length,
        "stored_tokens_per_sequence": profile.context_length + 1,
        "sequences_per_block": profile.sequences_per_block,
        "target_shard_bytes": profile.target_shard_bytes,
        "accepted_source_tokens": accepted,
        "validation_source_tokens": validation_source,
        "production": {
            "run_id": profile.run_id or "20m-qualification-dataset-001",
            "configuration_hash": "b" * 64,
            "schema_hash": "c" * 64,
            "target_source_tokens": profile.target_source_tokens,
            "minimum_source_tokens": profile.minimum_source_tokens,
            "maximum_source_tokens": profile.maximum_source_tokens,
            "checkpoint_source_tokens": profile.checkpoint_source_tokens,
            "target_reached": True,
            "remote_required": True,
        },
        "shards": [
            _shard("train", 0, train_blocks - 1, train_sequences),
            _shard(
                "validation",
                0,
                validation_blocks - 1,
                validation_blocks * profile.sequences_per_block,
            ),
        ],
    }


def _drive_manifest(manifest: dict[str, object]) -> dict[str, object]:
    production = manifest["production"]
    assert isinstance(production, dict)
    shards = manifest["shards"]
    assert isinstance(shards, list)
    return {
        "version": 1,
        "run_id": production["run_id"],
        "configuration_hash": production["configuration_hash"],
        "schema_hash": production["schema_hash"],
        "shards": [
            {
                "filename": shard["filename"],
                "drive_file_id": f"drive-{index}",
                "byte_size": shard["byte_size"],
                "local_sha256": shard["checksum"],
                "remote_durable": True,
                "configuration_hash": production["configuration_hash"],
                "schema_hash": production["schema_hash"],
            }
            for index, shard in enumerate(shards)
            if isinstance(shard, dict)
        ],
    }


class DatasetQualificationTests(unittest.TestCase):
    def test_registry_is_single_source_for_finite_dataset_contracts(self) -> None:
        expected = {
            "20m-10m": (None, 10_000_000, 9_000_000, 11_000_000, 2_000_000, False),
            "20m-100m": ("20m-100m-dataset-001", 100_000_000, 90_000_000, 110_000_000, 20_000_000, True),
            "20m-500m": ("20m-500m-dataset-001", 500_000_000, 450_000_000, 550_000_000, 20_000_000, True),
            "20m-2b": ("20m-2b-dataset-001", 2_000_000_000, 1_800_000_000, 2_200_000_000, 80_000_000, True),
            "modal-2b-b64": ("modal-2b-b64-dataset-001", 2_000_000_000, 1_800_000_000, 2_200_000_000, 80_000_000, True),
            "modal-10b-b64": ("modal-10b-b64-dataset-001", 10_000_000_000, 9_000_000_000, 11_000_000_000, 500_000_000, True),
        }
        self.assertEqual(set(PROFILES), set(expected))
        for key, values in expected.items():
            profile = get_profile(key)
            self.assertEqual(
                (
                    profile.run_id,
                    profile.target_source_tokens,
                    profile.minimum_source_tokens,
                    profile.maximum_source_tokens,
                    profile.checkpoint_source_tokens,
                    profile.production_enabled,
                ),
                values,
            )
            self.assertEqual(profile.context_length, 2048)
            if key == "modal-2b-b64":
                self.assertEqual(profile.sequences_per_block, 64)
                self.assertEqual(profile.target_shard_bytes, 32 * 1024 * 1024)
            elif key == "modal-10b-b64":
                self.assertEqual(profile.sequences_per_block, 64)
                self.assertEqual(profile.target_shard_bytes, 1024**3)
                self.assertEqual(profile.remote_backend, "hf_bucket")
                self.assertTrue(profile.evict_remote_shards)
            else:
                self.assertEqual(profile.sequences_per_block, 16)
                self.assertEqual(profile.target_shard_bytes, 8_388_608)

    def test_short_budget_aliases_resolve_to_canonical_profiles(self) -> None:
        aliases = (
            ("10m", "20m-10m"),
            ("100m", "20m-100m"),
            ("500m", "20m-500m"),
            ("2b", "20m-2b"),
            ("modal-2b", "modal-2b-b64"),
            ("10b", "modal-10b-b64"),
            ("modal-10b", "modal-10b-b64"),
        )
        for alias, canonical in aliases:
            with self.subTest(alias=alias):
                self.assertIs(get_profile(alias), get_profile(canonical))

    def test_active_profiles_append_exact_locked_production_identity(self) -> None:
        for key in ("20m-100m", "20m-500m", "20m-2b", "modal-2b-b64", "modal-10b-b64"):
            with self.subTest(profile=key):
                profile = get_profile(key)
                args = production_arguments(
                    key,
                    ["--weights-file", "weights.json", "--output-dir", "out"],
                )
                expected = {
                    "--run-id": profile.run_id,
                    "--target-tokens": str(profile.target_source_tokens),
                    "--minimum-tokens": str(profile.minimum_source_tokens),
                    "--maximum-tokens": str(profile.maximum_source_tokens),
                    "--checkpoint-source-tokens": str(profile.checkpoint_source_tokens),
                    "--context-length": str(profile.context_length),
                    "--sequences-per-block": str(profile.sequences_per_block),
                    "--target-shard-bytes": str(profile.target_shard_bytes),
                    "--remote-backend": profile.remote_backend,
                }
                for flag, value in expected.items():
                    index = args.index(flag)
                    self.assertEqual(args[index + 1], value)
                self.assertEqual("--evict-remote-shards" in args, profile.evict_remote_shards)

    def test_identity_and_geometry_overrides_are_rejected(self) -> None:
        for flag in ("--run-id", "--target-tokens", "--checkpoint-source-tokens", "--sequences-per-block", "--allow-local-only", "--remote-backend", "--evict-remote-shards"):
            with self.subTest(flag=flag), self.assertRaisesRegex(SystemExit, "fixes these arguments"):
                production_arguments(
                    "20m-2b",
                    [flag, "wrong"] if flag not in {"--allow-local-only", "--evict-remote-shards"} else [flag],
                )

    def test_historical_10m_profile_cannot_be_produced_again(self) -> None:
        with self.assertRaisesRegex(SystemExit, "historical"):
            production_arguments("20m-10m", ["--weights-file", "weights.json"])

    def test_build_cli_delegates_to_shared_producer(self) -> None:
        with patch("dataset.qualification.production_main", return_value=7) as producer:
            result = main(
                [
                    "build",
                    "--profile",
                    "2b",
                    "--weights-file",
                    "weights.json",
                    "--output-dir",
                    "out",
                    "--reader-workers",
                    "8",
                ]
            )
        self.assertEqual(result, 7)
        delegated = producer.call_args.args[0]
        self.assertIn("--reader-workers", delegated)
        self.assertIn("--run-id", delegated)
        self.assertIn("20m-2b-dataset-001", delegated)
        self.assertIn("--remote-backend", delegated)

    def test_exact_wsd_schedules_are_preserved(self) -> None:
        expected = {
            "20m-100m": (3_052, 153, 2_288, 611, 5_013_504, 74_973_184, 20_021_248, 100_007_936),
            "20m-500m": (15_250, 763, 11_437, 3_050, 25_001_984, 374_767_616, 99_942_400, 499_712_000),
            "20m-2b": (61_035, 3_052, 45_776, 12_207, 100_007_936, 1_499_987_968, 399_998_976, 1_999_994_880),
            "modal-2b-b64": (15_259, 763, 11_444, 3_052, 100_007_936, 1_499_987_968, 399_998_976, 1_999_994_880),
        }
        for key, values in expected.items():
            with self.subTest(profile=key):
                steps, warmup, stable, decay, warmup_tokens, stable_tokens, decay_tokens, total_tokens = values
                manifest = _manifest(
                    key,
                    steps,
                    train_sequences=976_560 if key == "modal-2b-b64" else None,
                )
                plan = derive_plan(manifest, profile=key)
                trainer = plan["trainer"]
                self.assertEqual(trainer["steps"], steps)
                self.assertEqual(trainer["warmup_updates"], warmup)
                self.assertEqual(trainer["stable_updates"], stable)
                self.assertEqual(trainer["decay_updates"], decay)
                self.assertEqual(trainer["warmup_tokens"], warmup_tokens)
                self.assertEqual(trainer["stable_tokens"], stable_tokens)
                self.assertEqual(trainer["decay_tokens"], decay_tokens)
                self.assertEqual(plan["train"]["target_tokens"], total_tokens)

    def test_historical_10m_plan_remains_reproducible(self) -> None:
        plan = derive_plan(_manifest("20m-10m", 305, validation_blocks=1), profile="20m-10m")
        trainer = plan["trainer"]
        self.assertEqual(plan["qualification_profile"], "20m-one-pass-v1")
        self.assertEqual(trainer["steps"], 305)
        self.assertEqual(trainer["warmup_updates"], 16)
        self.assertEqual(trainer["stable_updates"], 228)
        self.assertEqual(trainer["decay_updates"], 61)

    def test_selected_profile_binds_run_id(self) -> None:
        manifest = _manifest("20m-2b", 61_035)
        production = manifest["production"]
        assert isinstance(production, dict)
        production["run_id"] = "20m-1b-dataset-001"
        with self.assertRaisesRegex(ValueError, "run_id mismatch"):
            derive_plan(manifest, profile="20m-2b")

    def test_drive_manifest_is_bound_into_identity(self) -> None:
        manifest = _manifest("20m-100m", 3_052)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drive_manifest.json"
            path.write_text(json.dumps(_drive_manifest(manifest)), encoding="utf-8")
            plan = derive_plan(manifest, profile="20m-100m", drive_manifest_path=path)
        self.assertEqual(plan["identity"]["drive_run_id"], "20m-100m-dataset-001")
        self.assertEqual(plan["identity"]["drive_shard_count"], 2)


if __name__ == "__main__":
    unittest.main()
