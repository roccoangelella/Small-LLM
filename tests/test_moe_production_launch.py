"""Static and CPU qualification for the accepted MoE provider launch path."""

from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from moe_production import (
    ProductionRequest,
    accepted_identity,
    build_training_command,
    prepare_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


class TestProductionCommand(unittest.TestCase):
    def _request(self, **overrides) -> ProductionRequest:
        values = {
            "run_id": "moe-prod-qualification",
            "dataset_dir": "/data/moe-superbpe",
            "total_steps": 2,
            "precision": "fp16",
            "microbatch_size": 8,
            "source_commit": "a" * 40,
            "checkpoint_every_steps": 1,
        }
        values.update(overrides)
        return ProductionRequest(**values)

    def test_command_freezes_the_accepted_identity(self) -> None:
        command = build_training_command(self._request(), run_root=Path("/runs"))

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(value("--model-size"), "accepted")
        self.assertEqual(value("--architecture"), "gdn2_hybrid")
        self.assertEqual(value("--gdn-chunk-size"), "32")
        self.assertEqual(value("--load-balancing"), "quantile")
        self.assertEqual(value("--balancing-step-size"), "0")
        self.assertNotIn("substantive", command)

        identity = accepted_identity()
        self.assertEqual(
            (identity["num_experts"], identity["top_k"], identity["expert_d_ff"]),
            (64, 2, 352),
        )
        self.assertEqual(
            (identity["semantic_vocab_size"], identity["padded_vocab_size"]),
            (8_000, 8_192),
        )
        self.assertEqual(identity["load_balancing"], "quantile")

    def test_historical_pilot_launchers_remain_separate(self) -> None:
        for provider, gpu in (("modal", 'gpu="H100"'), ("beam", 'gpu="RTX4090"')):
            source = (ROOT / provider / "moe_production_launch.py").read_text(encoding="utf-8")
            self.assertIn(gpu, source)
            self.assertIn("run_provider_payload", source)
            self.assertNotIn("import moe_pilot", source)


class TestProductionDatasetGate(unittest.TestCase):
    def _dataset(self, root: Path, *, final_token: int) -> Path:
        (root / "train").mkdir(parents=True)
        values = array("H", [0] * 2_048 + [final_token])
        if sys.byteorder != "little":
            values.byteswap()
        payload = values.tobytes()
        shard = root / "train" / "000.bin"
        shard.write_bytes(payload)
        manifest = {
            "schema_version": 2,
            "sequence_format": "context_plus_one",
            "context_length": 2_048,
            "stored_tokens_per_sequence": 2_049,
            "sequences_per_block": 1,
            "shards": [
                {
                    "filename": "train/000.bin",
                    "split": "train",
                    "byte_size": len(payload),
                    "checksum": hashlib.sha256(payload).hexdigest(),
                    "sequence_count": 1,
                    "first_block_id": 0,
                    "last_block_id": 0,
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def _request(self, dataset: Path) -> ProductionRequest:
        return ProductionRequest(
            run_id="vocab-gate",
            dataset_dir=str(dataset),
            total_steps=1,
            precision="fp32",
            microbatch_size=1,
            source_commit="b" * 40,
            sequences_per_block=1,
        )

    def test_id_7999_is_a_valid_semantic_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary), final_token=7_999)
            prepared = prepare_dataset(self._request(dataset))
            self.assertEqual(prepared["status"], "ready")
            self.assertEqual(prepared["identity"]["semantic_vocab_size"], 8_000)

    def test_id_8000_is_rejected_even_though_embedding_storage_is_8192(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary), final_token=8_000)
            with self.assertRaisesRegex(ValueError, "outside the semantic vocabulary"):
                prepare_dataset(self._request(dataset))


if __name__ == "__main__":
    unittest.main()
