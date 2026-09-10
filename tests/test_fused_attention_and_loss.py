"""CPU contracts: fused SDPA attention and chunked cross-entropy equal the
unfused reference paths (values and gradients)."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from model.components import GatedMultiheadAttention
from model.config import ModelConfig
from model.fused_loss import chunked_cross_entropy_sum
from model.model import SmallLLM
from MOE_model.config import MoEModelConfig
from MOE_model.model import MoESmallLLM
from trainer.types import IGNORE_INDEX


def _dense_config(**overrides):
    values = {
        "semantic_vocab_size": 24,
        "padded_vocab_size": 32,
        "max_seq_len": 16,
        "d_model": 32,
        "n_layers": 4,
        "d_ff": 64,
        "n_heads": 2,
        "head_dim": 16,
        "gdn_num_key_heads": 2,
        "gdn_num_value_heads": 2,
        "gdn_key_dim": 16,
        "gdn_value_dim": 16,
        "layer_pattern": ("gdn", "gdn", "gdn", "mha"),
    }
    values.update(overrides)
    return ModelConfig(**values)


def _reference_attention(module: GatedMultiheadAttention, x: torch.Tensor) -> torch.Tensor:
    config = module.config
    batch, sequence, _ = x.shape
    q = module.q_proj(x).view(batch, sequence, config.n_heads, config.head_dim)
    k = module.k_proj(x).view(batch, sequence, config.n_heads, config.head_dim)
    v = module.v_proj(x).view(batch, sequence, config.n_heads, config.head_dim)
    q = module.rotary(module.q_norm(q))
    k = module.rotary(module.k_norm(k))
    mixed = module.reference_mix(q, k, v).reshape(batch, sequence, config.d_model)
    return module.out_proj(mixed * torch.sigmoid(module.gate_proj(x)))


class FusedAttentionTests(unittest.TestCase):
    def _check(self, window: int | None) -> None:
        torch.manual_seed(3)
        fused = GatedMultiheadAttention(_dense_config(attention_window=window))
        reference = copy.deepcopy(fused)
        x = torch.randn(2, 11, 32)
        x_fused = x.clone().requires_grad_(True)
        x_reference = x.clone().requires_grad_(True)
        out_fused = fused(x_fused)
        out_reference = _reference_attention(reference, x_reference)
        torch.testing.assert_close(out_fused, out_reference, rtol=1e-5, atol=1e-6)
        out_fused.square().sum().backward()
        out_reference.square().sum().backward()
        torch.testing.assert_close(x_fused.grad, x_reference.grad, rtol=1e-5, atol=1e-6)
        for (name, p_fused), (_, p_reference) in zip(
            fused.named_parameters(), reference.named_parameters()
        ):
            torch.testing.assert_close(p_fused.grad, p_reference.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_full_causal_attention_matches_reference(self) -> None:
        self._check(window=None)

    def test_sliding_window_attention_matches_reference(self) -> None:
        self._check(window=4)

    def test_causality_and_window_semantics_are_preserved(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(1, 8, 32)
        full = GatedMultiheadAttention(_dense_config())
        changed = x.clone()
        changed[:, 7] += 100
        torch.testing.assert_close(full(x)[:, :7], full(changed)[:, :7], rtol=1e-5, atol=1e-5)
        window = GatedMultiheadAttention(_dense_config(attention_window=2))
        window_changed = x.clone()
        window_changed[:, 2] += 100
        torch.testing.assert_close(window(x)[:, 7], window(window_changed)[:, 7], rtol=1e-5, atol=1e-5)


class ChunkedCrossEntropyTests(unittest.TestCase):
    def _case(self, chunk_tokens: int, with_ignore: bool) -> None:
        torch.manual_seed(5)
        tokens, d_model, semantic, padded = 37, 16, 24, 32
        hidden = torch.randn(3, 13, d_model)[:, :, :]
        hidden = hidden.reshape(-1, d_model)[:tokens].reshape(1, tokens, d_model)
        weight = torch.randn(padded, d_model)
        with torch.no_grad():
            weight[semantic:].zero_()
        labels = torch.randint(0, semantic, (1, tokens))
        if with_ignore:
            labels[0, ::5] = IGNORE_INDEX
        h_full = hidden.clone().requires_grad_(True)
        w_full = weight.clone().requires_grad_(True)
        logits = F.linear(h_full, w_full)[..., :semantic]
        full = F.cross_entropy(
            logits.reshape(-1, semantic), labels.reshape(-1), reduction="sum", ignore_index=IGNORE_INDEX
        )
        h_chunk = hidden.clone().requires_grad_(True)
        w_chunk = weight.clone().requires_grad_(True)
        chunked = chunked_cross_entropy_sum(
            h_chunk,
            w_chunk,
            labels,
            semantic_vocab_size=semantic,
            ignore_index=IGNORE_INDEX,
            chunk_tokens=chunk_tokens,
        )
        torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-5)
        full.backward()
        chunked.backward()
        torch.testing.assert_close(h_chunk.grad, h_full.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(w_chunk.grad, w_full.grad, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(w_chunk.grad[semantic:], torch.zeros(padded - semantic, d_model)))

    def test_matches_full_logits_for_divisible_and_ragged_chunks(self) -> None:
        for chunk in (37, 8, 5, 1000):
            self._case(chunk, with_ignore=False)

    def test_ignore_index_positions_contribute_nothing(self) -> None:
        for chunk in (37, 7):
            self._case(chunk, with_ignore=True)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            chunked_cross_entropy_sum(
                torch.randn(1, 4, 8), torch.randn(10, 8), torch.zeros(1, 5, dtype=torch.long),
                semantic_vocab_size=10, ignore_index=IGNORE_INDEX,
            )


class EndToEndTrainingLossTests(unittest.TestCase):
    def test_dense_model_chunked_loss_and_gradients_match_full_logits(self) -> None:
        torch.manual_seed(7)
        model = SmallLLM(_dense_config())
        reference = copy.deepcopy(model)
        ids = torch.randint(0, 24, (2, 9))
        labels = torch.randint(0, 24, (2, 9))
        labels[1, -2:] = IGNORE_INDEX
        full_logits = reference(ids)
        full = F.cross_entropy(full_logits.reshape(-1, 24), labels.reshape(-1), reduction="sum")
        chunked = chunked_cross_entropy_sum(
            model.hidden_states(ids), model.token_embedding.weight, labels,
            semantic_vocab_size=24, ignore_index=IGNORE_INDEX, chunk_tokens=4,
        )
        torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-5)
        full.backward()
        chunked.backward()
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            if q.grad is None:
                self.assertIsNone(p.grad, name)
                continue
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_moe_model_chunked_loss_and_gradients_match_full_logits(self) -> None:
        torch.manual_seed(11)
        dense = ModelConfig(
            semantic_vocab_size=32, padded_vocab_size=32, max_seq_len=8, d_model=64, n_layers=4,
            d_ff=96, n_heads=4, head_dim=16, gdn_num_key_heads=4, gdn_num_value_heads=4,
            gdn_key_dim=16, gdn_value_dim=16, gdn_conv_kernel_size=4, gdn_chunk_size=4,
        )
        config = MoEModelConfig(dense=dense, expert_d_ff=96, moe_layer_indices=tuple(range(4)))
        model = MoESmallLLM(config)
        reference = copy.deepcopy(model)
        ids = torch.randint(0, 32, (3, 8))
        labels = torch.randint(0, 32, (3, 8))
        full_logits, aux_ref = reference.forward_with_aux(ids)
        full = F.cross_entropy(full_logits.reshape(-1, 32), labels.reshape(-1), reduction="sum")
        hidden, aux = model.hidden_states_with_aux(ids)
        chunked = chunked_cross_entropy_sum(
            hidden, model.token_embedding.weight, labels,
            semantic_vocab_size=32, ignore_index=IGNORE_INDEX, chunk_tokens=6,
        )
        torch.testing.assert_close(chunked, full, rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(aux.z_loss, aux_ref.z_loss)
        (full + 1e-4 * aux_ref.z_loss).backward()
        (chunked + 1e-4 * aux.z_loss).backward()
        for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
            if q.grad is None:
                self.assertIsNone(p.grad, name)
                continue
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)


if __name__ == "__main__":
    unittest.main()
