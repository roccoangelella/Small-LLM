from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from model.config import ModelConfig
from model.initialization import initialize_model
from model.model import SmallLLM


RSFT_DIR = Path(__file__).resolve().parents[1] / "post_training" / "R-SFT"


def _load_transition():
    module_name = "small_llm_rsft_model_transition_test"
    path = RSFT_DIR / "model_transition.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


transition = _load_transition()


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        semantic_vocab_size=50_257,
        padded_vocab_size=50_304,
        max_seq_len=64,
        d_model=64,
        n_layers=4,
        d_ff=128,
        n_heads=1,
        head_dim=64,
        gdn_num_key_heads=1,
        gdn_num_value_heads=1,
        gdn_key_dim=64,
        gdn_value_dim=64,
    )


def _parent() -> tuple[SmallLLM, ModelConfig]:
    config = _tiny_config()
    torch.manual_seed(123)
    model = SmallLLM(config)
    initialize_model(model, "normal")
    return model, config


class ReasoningSFTModelTransitionTests(unittest.TestCase):
    def test_promotion_preserves_every_parent_parameter_except_three_rows(self) -> None:
        parent, config = _parent()
        parent_state = {name: value.detach().clone() for name, value in parent.state_dict().items()}

        promoted, promoted_config = transition.promote_s0_model_for_rsft(
            parent,
            config,
            seed=17,
        )

        self.assertEqual(promoted_config.semantic_vocab_size, 50_260)
        self.assertEqual(promoted_config.padded_vocab_size, config.padded_vocab_size)
        self.assertEqual(
            promoted.token_embedding.weight.shape,
            parent.token_embedding.weight.shape,
        )

        promoted_state = promoted.state_dict()
        for name, before in parent_state.items():
            after = promoted_state[name]
            if name == "token_embedding.weight":
                self.assertTrue(torch.equal(after[:50_257], before[:50_257]))
                self.assertGreater(int(torch.count_nonzero(after[50_257:50_260]).item()), 0)
                self.assertEqual(int(torch.count_nonzero(after[50_260:]).item()), 0)
            else:
                self.assertTrue(torch.equal(after, before), name)

        # The parent itself remains an ordinary S0 model with zero padding.
        self.assertEqual(parent.config.semantic_vocab_size, 50_257)
        self.assertEqual(int(torch.count_nonzero(parent.token_embedding.weight[50_257:]).item()), 0)

    def test_promotion_is_deterministic_and_does_not_advance_global_rng(self) -> None:
        parent, config = _parent()
        torch.manual_seed(999)
        before = torch.random.get_rng_state().clone()
        first, _ = transition.promote_s0_model_for_rsft(parent, config, seed=17)
        after = torch.random.get_rng_state().clone()
        second, _ = transition.promote_s0_model_for_rsft(parent, config, seed=17)
        third, _ = transition.promote_s0_model_for_rsft(parent, config, seed=18)

        self.assertTrue(torch.equal(before, after))
        self.assertTrue(
            torch.equal(
                first.token_embedding.weight[50_257:50_260],
                second.token_embedding.weight[50_257:50_260],
            )
        )
        self.assertFalse(
            torch.equal(
                first.token_embedding.weight[50_257:50_260],
                third.token_embedding.weight[50_257:50_260],
            )
        )

    def test_promotion_rejects_non_s0_semantic_vocab(self) -> None:
        config = _tiny_config()
        parent = SmallLLM(config)
        wrong = ModelConfig(
            semantic_vocab_size=50_256,
            padded_vocab_size=config.padded_vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            gdn_num_key_heads=config.gdn_num_key_heads,
            gdn_num_value_heads=config.gdn_num_value_heads,
            gdn_key_dim=config.gdn_key_dim,
            gdn_value_dim=config.gdn_value_dim,
        )
        with self.assertRaisesRegex(ValueError, "requires an S0 semantic vocabulary"):
            transition.promote_s0_model_for_rsft(parent, wrong, seed=17)


if __name__ == "__main__":
    unittest.main()
