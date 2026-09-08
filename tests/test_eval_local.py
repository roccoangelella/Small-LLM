from __future__ import annotations

from array import array
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from MOE_model.config import MoEModelConfig
from MOE_model.model import MoESmallLLM
from model.config import ModelConfig
from model.model import SmallLLM
from trainer import eval_suite
from trainer import eval_local
from tests.test_eval_suite import _write_tiny_eval


def _config() -> MoEModelConfig:
    dense = ModelConfig(
        semantic_vocab_size=4,
        padded_vocab_size=8,
        max_seq_len=2_048,
        d_model=64,
        n_layers=4,
        d_ff=96,
        n_heads=4,
        head_dim=16,
        gdn_num_key_heads=4,
        gdn_num_value_heads=4,
        gdn_key_dim=16,
        gdn_value_dim=16,
        gdn_conv_kernel_size=4,
        gdn_chunk_size=64,
    )
    return MoEModelConfig(dense=dense, expert_d_ff=96, moe_layer_indices=(0, 1, 2, 3))


def _checkpoint(root: Path, model: torch.nn.Module, model_config: dict[str, object]) -> Path:
    root.mkdir()
    with (root / "trainer_state.pkl").open("wb") as handle:
        pickle.dump(
            {
                "version": 1,
                "model": model.state_dict(),
                "model_config": model_config,
                "global_step": 3,
                "consumed_tokens": 12,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    (root / "checkpoint.json").write_text(
        json.dumps(
            {
                "version": 1,
                "checkpoint_id": "step-00000003",
                "configuration_hash": "config",
                "source_hash": "source",
                "schema_hash": "schema",
                "optimizer_step_complete": True,
                "pipeline_state": {},
            }
        ),
        encoding="utf-8",
    )
    (root / "local_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"name": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                    for name in ("trainer_state.pkl", "checkpoint.json")
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


class LocalEvaluationTests(unittest.TestCase):
    def test_cli_wires_local_verification_loader_and_existing_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "checkpoint.json").write_text('{"version": 1, "optimizer_step_complete": true}', encoding="utf-8")
            (checkpoint / "local_manifest.json").write_text("{}", encoding="utf-8")
            output = root / "result.json"
            config = type("Config", (), {"max_seq_len": 2_048, "as_dict": lambda self: {"test": True}})()
            model = torch.nn.Linear(1, 1)
            with (
                patch.object(eval_local, "verify_local_manifest"),
                patch.object(eval_local, "load_moe_model", return_value=(model, config, {"global_step": 2})),
                patch.object(eval_local, "evaluate_split", return_value={"eval_manifest_sha256": "eval"}) as score,
            ):
                self.assertEqual(
                    eval_local.main(
                        [
                            "--checkpoint-dir", str(checkpoint), "--eval-dir", str(root / "eval"),
                            "--output-json", str(output), "--device", "cpu", "--moe",
                            "--batch-size", "4", "--bootstrap-samples", "7",
                        ]
                    ),
                    0,
                )
            score.assert_called_once()
            self.assertEqual(score.call_args.kwargs["suite"], "fast")
            self.assertEqual(score.call_args.kwargs["batch_size"], 4)
            self.assertEqual(json.loads(output.read_text())["eval_core_v1"]["eval_manifest_sha256"], "eval")

    def test_real_moe_checkpoint_scores_tiny_eval_core(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config()
            checkpoint = _checkpoint(root / "checkpoint", MoESmallLLM(config), config.as_dict())
            eval_dir = root / "eval_core_v1"
            eval_dir.mkdir()
            _write_tiny_eval(eval_dir)
            real_verify = eval_suite.verify_eval_core
            with (
                patch.object(eval_suite, "verify_eval_core",
                             side_effect=lambda path, **_: real_verify(path, enforce_frozen_minimums=False)),
                patch.object(eval_suite, "_gpt2_token_byte_lengths", return_value=[1, 1, 1, 1]),
            ):
                eval_local.main([
                    "--checkpoint-dir", str(checkpoint), "--eval-dir", str(eval_dir),
                    "--output-json", str(root / "result.json"), "--device", "cpu",
                    "--precision", "fp32", "--batch-size", "19", "--bootstrap-samples", "2", "--moe",
                ])
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["model_config"], json.loads(json.dumps(config.as_dict())))
            self.assertEqual(result["eval_core_v1"]["target_tokens"], 3 * 19)
            self.assertEqual(result["eval_core_v1"]["eval_manifest_sha256"], json.loads((eval_dir / "manifest.json").read_text())["manifest_sha256"])

    def test_real_dense_checkpoint_scores_tiny_eval_core(self) -> None:
        # The dense ModelConfig is a plain dataclass without as_dict(); the adapter must still write model_config.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense = _config().dense
            checkpoint = _checkpoint(root / "checkpoint", SmallLLM(dense), asdict(dense))
            eval_dir = root / "eval_core_v1"
            eval_dir.mkdir()
            _write_tiny_eval(eval_dir)
            real_verify = eval_suite.verify_eval_core
            with (
                patch.object(eval_suite, "verify_eval_core",
                             side_effect=lambda path, **_: real_verify(path, enforce_frozen_minimums=False)),
                patch.object(eval_suite, "_gpt2_token_byte_lengths", return_value=[1, 1, 1, 1]),
            ):
                eval_local.main([
                    "--checkpoint-dir", str(checkpoint), "--eval-dir", str(eval_dir),
                    "--output-json", str(root / "result.json"), "--device", "cpu",
                    "--precision", "fp32", "--batch-size", "19", "--bootstrap-samples", "2",
                ])
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["model_config"], json.loads(json.dumps(asdict(dense))))
            self.assertEqual(result["eval_core_v1"]["target_tokens"], 3 * 19)


if __name__ == "__main__":
    unittest.main()
