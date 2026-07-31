import unittest

import torch

from model.accounting import count_parameters, gdn_exception_counts, optimizer_no_weight_decay_parameter_names
from model.components import GatedMultiheadAttention, RMSNorm, RotaryEmbedding, SwiGLU, TiedEmbedding
from model.config import ModelConfig
from model.initialization import compare_initializations, initialize_model
from model.model import SmallLLM


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
        "layer_pattern": ("gdn", "gdn", "gdn", "mha"),
    }
    values.update(overrides)
    return ModelConfig(**values)


class SharedComponentTests(unittest.TestCase):
    def test_config_factories_and_validation(self):
        self.assertEqual(ModelConfig.smoke().layer_kinds, ("gdn", "gdn", "gdn", "mha") * 2)
        self.assertEqual(ModelConfig.substantive().d_model, 512)
        self.assertIsNone(ModelConfig.smoke().attention_window)
        with self.assertRaises(ValueError):
            tiny_config(d_model=33)
        with self.assertRaises(ValueError):
            tiny_config(layer_pattern=("mha",) * 4)
        with self.assertRaises(ValueError):
            tiny_config(attention_window=9)
        with self.assertRaises(ValueError):
            tiny_config(dropout=0.1)
        self.assertEqual(
            tiny_config(architecture="swa_hybrid", max_seq_len=512).layer_kinds,
            ("swa", "swa", "swa", "mha"),
        )
        with self.assertRaises(ValueError):
            tiny_config(architecture="swa_hybrid", attention_window=256, max_seq_len=512)
        with self.assertRaises(ValueError):
            RMSNorm(8, float("nan"))
        with self.assertRaises(ValueError):
            RotaryEmbedding(8, float("nan"))

    def test_rmsnorm_rope_and_swiglu(self):
        x = torch.randn(2, 3, 32, requires_grad=True)
        self.assertEqual(RMSNorm(32)(x).shape, x.shape)
        rotary = RotaryEmbedding(16)
        q = torch.randn(2, 3, 2, 16)
        self.assertTrue(torch.allclose(rotary(q).square().sum(-1), q.square().sum(-1), atol=1e-5))
        result = SwiGLU(32, 64)(x)
        result.square().mean().backward()
        self.assertEqual(result.shape, x.shape)
        self.assertIsNotNone(x.grad)

    def test_tied_embedding_masks_padding_from_language_model(self):
        embedding = TiedEmbedding(tiny_config())
        self.assertEqual(embedding.weight.shape, (32, 32))
        self.assertTrue(torch.equal(embedding.weight[24:], torch.zeros_like(embedding.weight[24:])))
        logits = embedding.logits(embedding(torch.tensor([[0, 23]])))
        self.assertEqual(logits.shape, (1, 2, 24))
        with self.assertRaises(ValueError):
            embedding(torch.tensor([[24]]))

    def test_attention_is_causal_and_window_is_opt_in(self):
        torch.manual_seed(0)
        x = torch.randn(1, 6, 32)
        full = GatedMultiheadAttention(tiny_config())
        changed = x.clone()
        changed[:, 5] += 100
        self.assertTrue(torch.allclose(full(x)[:, :5], full(changed)[:, :5], atol=1e-5, rtol=1e-5))
        window = GatedMultiheadAttention(tiny_config(attention_window=2))
        self.assertEqual(window(x).shape, x.shape)
        window_changed = x.clone()
        window_changed[:, 1] += 100
        self.assertTrue(torch.allclose(window(x)[:, 5], window(window_changed)[:, 5], atol=1e-5, rtol=1e-5))
        window(x).square().mean().backward()

    def test_attention_cpu_autocast_stays_finite(self):
        attention = GatedMultiheadAttention(tiny_config())
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = attention(torch.randn(1, 4, 32))
        self.assertTrue(torch.isfinite(output).all())


class ModelAssemblyTests(unittest.TestCase):
    def test_all_mha_forward_backward_and_semantic_logits(self):
        model = SmallLLM(tiny_config(), all_mha=True)
        ids = torch.randint(0, 24, (2, 5))
        logits = model(ids)
        self.assertEqual(logits.shape, (2, 5, 24))
        logits.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_hybrid_pattern_final_norm_and_sequential_blocks(self):
        model = SmallLLM(tiny_config())
        self.assertEqual(model.layer_kinds, ("gdn", "gdn", "gdn", "mha"))
        self.assertIsInstance(model.final_norm, RMSNorm)
        x = torch.randn(1, 3, 32)
        block = model.blocks[0]
        mixed = block.mixer(block.mixer_norm(x))
        expected = x + mixed
        expected = expected + block.ffn(block.ffn_norm(expected))
        self.assertTrue(torch.allclose(block(x), expected))

    def test_plan_b_uses_three_swa_layers_then_full_attention(self):
        config = tiny_config(architecture="swa_hybrid", max_seq_len=512)
        model = SmallLLM(config)
        self.assertEqual(model.layer_kinds, ("swa", "swa", "swa", "mha"))
        self.assertEqual([block.mixer.config.attention_window for block in model.blocks], [512, 512, 512, None])
        self.assertEqual(model(torch.randint(0, 24, (1, 4))).shape, (1, 4, 24))

    def test_unqualified_plan_a5_refuses_to_substitute_gdn2(self):
        with self.assertRaises(NotImplementedError):
            SmallLLM(tiny_config(architecture="gdn_v1_hybrid"))

    def test_parameter_accounting_counts_tied_weight_once(self):
        model = SmallLLM(tiny_config(), all_mha=True)
        counts = count_parameters(model)
        self.assertEqual(counts.total, sum(parameter.numel() for parameter in model.parameters()))
        self.assertEqual(counts.embeddings, 32 * 32)
        self.assertEqual(counts.gdn_mixers, 0)
        self.assertGreater(counts.mha_mixers, 0)

    def test_all_mha_baseline_is_parameter_matched(self):
        hybrid = SmallLLM(tiny_config())
        baseline = SmallLLM(tiny_config(), all_mha=True)
        selected_plan_c = SmallLLM(tiny_config(architecture="all_mha"))
        hybrid_total = count_parameters(hybrid).total
        baseline_total = count_parameters(baseline).total
        self.assertNotEqual(baseline.ffn_width, hybrid.config.d_ff)
        self.assertLess(abs(hybrid_total - baseline_total) / hybrid_total, 0.003)
        self.assertEqual(selected_plan_c.ffn_width, baseline.ffn_width)
        self.assertEqual(count_parameters(selected_plan_c).total, baseline_total)

    def test_gdn_exceptions_and_decay_exclusions_are_named(self):
        model = SmallLLM(tiny_config())
        exceptions = gdn_exception_counts(model)
        self.assertEqual(exceptions.a_log, 6)
        self.assertEqual(exceptions.dt_bias, 96)
        self.assertEqual(exceptions.output_gate_bias, 96)
        exclusions = optimizer_no_weight_decay_parameter_names(model)
        self.assertIn("blocks.0.mixer.A_log", exclusions)
        self.assertIn("blocks.0.mixer.dt_bias", exclusions)
        self.assertIn("blocks.0.mixer_norm.weight", exclusions)
        self.assertIn("blocks.0.mixer.output_norm.weight", exclusions)

    def test_initialization_experiment_keeps_contract_invariants(self):
        model = SmallLLM(tiny_config())
        initial_a_log = model.blocks[0].mixer.A_log.detach().clone()
        initial_dt_bias = model.blocks[0].mixer.dt_bias.detach().clone()
        initialize_model(model, "xavier")
        self.assertTrue(torch.equal(model.token_embedding.weight[24:], torch.zeros_like(model.token_embedding.weight[24:])))
        self.assertTrue(torch.equal(model.blocks[0].mixer.A_log, initial_a_log))
        self.assertTrue(torch.equal(model.blocks[0].mixer.dt_bias, initial_dt_bias))
        metrics = compare_initializations(model, torch.zeros(1, 3, dtype=torch.long))
        self.assertEqual(set(metrics), {"normal", "xavier"})
        self.assertTrue(all(values["finite"] and not values["overflow"] for values in metrics.values()))

    def test_initialization_comparison_preserves_global_rng(self):
        model = SmallLLM(tiny_config(), all_mha=True)
        torch.manual_seed(123)
        before = torch.random.get_rng_state()
        compare_initializations(model)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        before_factory = torch.random.get_rng_state()
        compare_initializations(lambda: SmallLLM(tiny_config(), all_mha=True))
        self.assertTrue(torch.equal(before_factory, torch.random.get_rng_state()))

    def test_tiny_deterministic_overfit_reduces_loss(self):
        torch.manual_seed(4)
        model = SmallLLM(tiny_config(), all_mha=True)
        initialize_model(model, "normal")
        ids = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 24), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
