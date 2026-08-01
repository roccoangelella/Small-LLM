import unittest

import torch

from model.config import ModelConfig
from model.gdn2 import (
    GDN2Cache,
    GatedDeltaNet2,
    OptimizedGDN2BackendAdapter,
    PyTorchChunkwiseGDN2Backend,
    PyTorchGDN2Backend,
    assert_gdn2_backend_parity,
    gdn2_chunkwise_reference,
    gdn2_recurrent_reference,
)


def tiny_config(**overrides):
    values = {
        "semantic_vocab_size": 24,
        "padded_vocab_size": 32,
        "max_seq_len": 8,
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
        "gdn_chunk_size": 3,
        "layer_pattern": ("gdn", "gdn", "gdn", "mha"),
    }
    values.update(overrides)
    return ModelConfig(**values)


class GDN2RecurrenceTests(unittest.TestCase):
    def test_reference_uses_updated_state_and_key_rows(self):
        q = torch.tensor([[[[1.0, 0.0]]]])
        k = torch.tensor([[[[1.0, 0.0]]]])
        v = torch.tensor([[[[3.0, 5.0]]]])
        zeros = torch.zeros_like(q)
        write = torch.ones_like(v)
        output, state = gdn2_recurrent_reference(q, k, v, zeros, zeros, write)
        expected_state = torch.tensor([[[[3.0, 5.0], [0.0, 0.0]]]])
        self.assertTrue(torch.equal(state, expected_state))
        self.assertTrue(
            torch.allclose(output, torch.tensor([[[[3.0 / 2**0.5, 5.0 / 2**0.5]]]]))
        )

    def test_chunkwise_outputs_states_and_gradients_match_oracle(self):
        torch.manual_seed(1)
        batch, sequence, heads, key_dim, value_dim = 1, 7, 2, 4, 3
        q = torch.randn(batch, sequence, heads, key_dim)
        k = torch.randn(batch, sequence, heads, key_dim)
        v = torch.randn(batch, sequence, heads, value_dim)
        log_decay = -0.05 * torch.rand(batch, sequence, heads, key_dim)
        erase = torch.sigmoid(torch.randn(batch, sequence, heads, key_dim))
        write = torch.sigmoid(torch.randn(batch, sequence, heads, value_dim))
        initial_state = torch.randn(batch, heads, key_dim, value_dim)

        backend = PyTorchChunkwiseGDN2Backend(chunk_size=3)
        assert_gdn2_backend_parity(
            backend,
            q,
            k,
            v,
            log_decay,
            erase,
            write,
            initial_state,
            atol=3e-5,
            rtol=3e-5,
            check_gradients=True,
        )
        direct_output, direct_state = gdn2_chunkwise_reference(
            q,
            k,
            v,
            log_decay,
            erase,
            write,
            initial_state,
            chunk_size=3,
        )
        backend_output, backend_state = backend(
            q, k, v, log_decay, erase, write, initial_state
        )
        self.assertTrue(torch.equal(direct_output, backend_output))
        self.assertTrue(torch.equal(direct_state, backend_state))

    def test_chunk_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            PyTorchChunkwiseGDN2Backend(0)
        q = k = v = torch.randn(1, 1, 1, 2)
        with self.assertRaises(ValueError):
            gdn2_chunkwise_reference(
                q,
                k,
                v,
                torch.zeros_like(q),
                torch.zeros_like(q),
                torch.zeros_like(v),
                chunk_size=0,
            )

    def test_adapter_without_backend_matches_reference(self):
        tensors = [torch.randn(1, 3, 1, 2) for _ in range(5)]
        q, k, v, decay, erase = tensors
        write = torch.sigmoid(v)
        reference = gdn2_recurrent_reference(q, k, v, decay, erase, write)
        adapted = OptimizedGDN2BackendAdapter()(q, k, v, decay, erase, write)
        self.assertTrue(torch.allclose(adapted[0], reference[0]))
        self.assertTrue(torch.allclose(adapted[1], reference[1]))

    def test_adapter_rejects_nonfinite_backend_results(self):
        def nonfinite_backend(q, k, v, decay, erase, write, state):
            return torch.full_like(v, float("nan")), torch.zeros(
                q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
            )

        q = k = v = torch.randn(1, 1, 1, 2)
        with self.assertRaises(ValueError):
            OptimizedGDN2BackendAdapter(nonfinite_backend)(
                q, k, v, torch.zeros_like(q), torch.zeros_like(q), torch.zeros_like(v)
            )

    def test_backend_parity_qualification_rejects_wrong_finite_results(self):
        q = k = v = torch.randn(1, 2, 1, 2)
        decay = erase = torch.zeros_like(q)
        write = torch.sigmoid(v)
        assert_gdn2_backend_parity(PyTorchGDN2Backend(), q, k, v, decay, erase, write)

        def wrong_backend(q, k, v, decay, erase, write, state):
            return torch.zeros_like(v), torch.zeros(
                q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
            )

        with self.assertRaises(AssertionError):
            assert_gdn2_backend_parity(wrong_backend, q, k, v, decay, erase, write)


class GDN2LayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.module = GatedDeltaNet2(tiny_config())

    def test_default_backend_uses_configured_chunk_size(self):
        self.assertIsInstance(self.module.backend, PyTorchChunkwiseGDN2Backend)
        self.assertEqual(self.module.backend.chunk_size, 3)

    def test_shape_backward_and_causality(self):
        x = torch.randn(2, 5, 32, requires_grad=True)
        output = self.module(x)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        changed = x.detach().clone()
        changed[:, 4] += 10
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(
                    self.module(x.detach())[:, :4],
                    self.module(changed)[:, :4],
                    atol=3e-5,
                    rtol=3e-5,
                )
            )

    def test_tokenwise_and_segmented_cache_match_one_shot(self):
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            full, full_cache = self.module(x, return_cache=True)
            pieces = []
            cache = None
            for token in x.split(1, dim=1):
                piece, cache = self.module(token, cache=cache, return_cache=True)
                pieces.append(piece)
            tokenwise = torch.cat(pieces, dim=1)
            first, cache = self.module(x[:, :2], return_cache=True)
            second, segmented_cache = self.module(x[:, 2:], cache=cache, return_cache=True)
        self.assertTrue(torch.allclose(tokenwise, full, atol=3e-5, rtol=3e-5))
        self.assertTrue(
            torch.allclose(torch.cat((first, second), dim=1), full, atol=3e-5, rtol=3e-5)
        )
        self.assertTrue(
            torch.allclose(
                segmented_cache.recurrent_state,
                full_cache.recurrent_state,
                atol=3e-5,
                rtol=3e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                segmented_cache.q_history, full_cache.q_history, atol=1e-5, rtol=1e-5
            )
        )

    def test_absent_cache_resets_state(self):
        x = torch.randn(1, 3, 32)
        with torch.no_grad():
            fresh_before = self.module(x[:, 1:])
            _, cache = self.module(x[:, :1], return_cache=True)
            cached_continuation = self.module(x[:, 1:], cache=cache)
            fresh_after = self.module(x[:, 1:])
        self.assertFalse(torch.allclose(cached_continuation, fresh_before))
        self.assertTrue(torch.allclose(fresh_before, fresh_after, atol=1e-6, rtol=1e-6))
        invalid = GDN2Cache(
            torch.full_like(cache.recurrent_state, float("nan")),
            cache.q_history,
            cache.k_history,
            cache.v_history,
        )
        with self.assertRaises(ValueError):
            self.module(x[:, :1], cache=invalid)
        wrong_state_dtype = GDN2Cache(
            cache.recurrent_state.double(), cache.q_history, cache.k_history, cache.v_history
        )
        with self.assertRaises(ValueError):
            self.module(x[:, :1], cache=wrong_state_dtype)
        wrong_history_dtype = GDN2Cache(
            cache.recurrent_state, cache.q_history.double(), cache.k_history, cache.v_history
        )
        with self.assertRaises(ValueError):
            self.module(x[:, :1], cache=wrong_history_dtype)


if __name__ == "__main__":
    unittest.main()
