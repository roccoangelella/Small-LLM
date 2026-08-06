import unittest

import torch
from torch.nn import functional as F

from model.config import ModelConfig
from model.gdn2 import GatedDeltaNet2, assert_gdn2_backend_parity
from model.gdn2_stable import AdaptiveChunkwiseGDN2Backend, StableGatedDeltaNet2
from model.model import SmallLLM


def tiny_config(**overrides):
    values = {
        "semantic_vocab_size": 24,
        "padded_vocab_size": 32,
        "max_seq_len": 32,
        "d_model": 32,
        "n_layers": 4,
        "d_ff": 64,
        "n_heads": 2,
        "head_dim": 16,
        "gdn_num_key_heads": 2,
        "gdn_num_value_heads": 2,
        "gdn_key_dim": 16,
        "gdn_value_dim": 16,
        "gdn_conv_kernel_size": 4,
        "gdn_chunk_size": 32,
        "layer_pattern": ("gdn", "gdn", "gdn", "mha"),
    }
    values.update(overrides)
    return ModelConfig(**values)


class AdaptiveChunkwiseGDN2Tests(unittest.TestCase):
    def test_strong_decay_matches_recurrent_oracle_with_gradients(self):
        torch.manual_seed(7)
        batch, sequence, heads, key_dim, value_dim = 1, 32, 2, 4, 3
        q = F.normalize(torch.randn(batch, sequence, heads, key_dim), dim=-1)
        k = F.normalize(torch.randn(batch, sequence, heads, key_dim), dim=-1)
        v = torch.randn(batch, sequence, heads, value_dim)
        log_decay = torch.full((batch, sequence, heads, key_dim), -6.0)
        erase = torch.sigmoid(torch.randn(batch, sequence, heads, key_dim))
        write = torch.sigmoid(torch.randn(batch, sequence, heads, value_dim))
        initial_state = torch.randn(batch, heads, key_dim, value_dim)

        assert_gdn2_backend_parity(
            AdaptiveChunkwiseGDN2Backend(chunk_size=32),
            q,
            k,
            v,
            log_decay,
            erase,
            write,
            initial_state,
            atol=5e-5,
            rtol=5e-5,
            check_gradients=True,
            gradient_atol=5e-5,
            gradient_rtol=5e-5,
        )

    def test_stable_layer_preserves_checkpoint_parameter_keys(self):
        config = tiny_config()
        legacy_keys = tuple(GatedDeltaNet2(config).state_dict())
        stable_keys = tuple(StableGatedDeltaNet2(config).state_dict())
        self.assertEqual(stable_keys, legacy_keys)

    def test_assembled_model_uses_adaptive_backend(self):
        model = SmallLLM(tiny_config())
        gdn_mixers = [
            block.mixer
            for kind, block in zip(model.layer_kinds, model.blocks, strict=True)
            if kind in {"gdn", "gdn-2"}
        ]
        self.assertTrue(gdn_mixers)
        self.assertTrue(all(isinstance(mixer, StableGatedDeltaNet2) for mixer in gdn_mixers))
        self.assertTrue(
            all(isinstance(mixer.backend, AdaptiveChunkwiseGDN2Backend) for mixer in gdn_mixers)
        )
        self.assertTrue(all(mixer.backend.chunk_size == 32 for mixer in gdn_mixers))


if __name__ == "__main__":
    unittest.main()
