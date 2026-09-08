"""MoE evaluation-loader identity tests."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import tempfile
import unittest

import torch

from MOE_model.config import MoEModelConfig
from MOE_model.evaluation import load_moe_model, normalize_moe_model_config
from MOE_model.model import MoESmallLLM
from model.config import ModelConfig


def _tiny_moe_config() -> MoEModelConfig:
    dense = ModelConfig(
        semantic_vocab_size=32,
        padded_vocab_size=32,
        max_seq_len=8,
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
        gdn_chunk_size=4,
    )
    return MoEModelConfig(
        dense=dense,
        expert_d_ff=96,
        moe_layer_indices=tuple(range(4)),
    )


class TestMoEEvaluation(unittest.TestCase):
    def test_checkpoint_model_config_round_trips_exactly(self) -> None:
        original = MoEModelConfig.smoke()
        payload = original.as_dict()
        # JSON checkpoint metadata may materialize tuples as lists.
        payload["moe_layer_indices"] = list(payload["moe_layer_indices"])
        dense = dict(payload["dense"])
        dense["layer_pattern"] = list(dense["layer_pattern"])
        payload["dense"] = dense

        restored = normalize_moe_model_config(payload)
        self.assertEqual(restored.as_dict(), original.as_dict())

    def test_eval_loader_rejects_dense_only_config(self) -> None:
        with self.assertRaises(RuntimeError):
            normalize_moe_model_config({"d_model": 512, "n_layers": 20})

    def test_eval_loader_fails_closed_on_topk_mismatch(self) -> None:
        payload = MoEModelConfig.smoke().as_dict()
        payload["top_k"] = 2
        with self.assertRaises(ValueError):
            normalize_moe_model_config(payload)

    def test_eval_loader_rejects_mismatched_config_override(self) -> None:
        config = _tiny_moe_config()
        model = MoESmallLLM(config)
        override = config.as_dict()
        dense = dict(override["dense"])
        dense["rms_norm_eps"] = 1e-3
        override["dense"] = dense

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            checkpoint.mkdir()
            with (checkpoint / "trainer_state.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "version": 1,
                        "model": model.state_dict(),
                        "model_config": config.as_dict(),
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            override_path = Path(temporary) / "override.json"
            override_path.write_text(json.dumps(override), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                load_moe_model(
                    checkpoint,
                    device=torch.device("cpu"),
                    model_config_json=override_path,
                )


if __name__ == "__main__":
    unittest.main()
