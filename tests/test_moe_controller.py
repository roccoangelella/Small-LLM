"""Controller, sparse AMP combine and committed-state contracts."""
from dataclasses import replace
import unittest

import torch

from MOE_model.config import MoEModelConfig
from MOE_model.engine import MoETrainerEngine
from MOE_model.model import DroplessTop1MoE, MoESmallLLM
from MOE_model.router import SwitchTop1Router
from tests.test_moe_model import _tiny_moe_config
from trainer import TrainerConfig
from trainer.engine import TrainerEngine
from tests.trainer_fixtures import TinyLM, batch


class TestMoEController(unittest.TestCase):
    def test_real_scaler_failed_update_preserves_model_optimizer_and_controller(self):
        def same_tree(first, second):
            if isinstance(first, torch.Tensor):
                self.assertTrue(torch.equal(first, second))
            elif isinstance(first, dict):
                self.assertEqual(first.keys(), second.keys())
                for key in first:
                    same_tree(first[key], second[key])
            elif isinstance(first, (list, tuple)):
                self.assertEqual(len(first), len(second))
                for a, b in zip(first, second, strict=True):
                    same_tree(a, b)
            else:
                self.assertEqual(first, second)

        config = TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32", microbatch_size=1, max_overflow_retries=1,
                               schedule="wsd", warmup_tokens=60, stable_tokens=60, decay_tokens=60)
        for kind in ("dense", "moe"):
            with self.subTest(kind=kind):
                torch.manual_seed(7)
                model = (TinyLM() if kind == "dense" else MoESmallLLM(replace(
                    _tiny_moe_config(), load_balancing="loss_free_sign", balancing_step_size=.001)))
                engine_cls = TrainerEngine if kind == "dense" else MoETrainerEngine
                engine = engine_cls(model, replace(config, optimizer=("adamw" if kind == "dense" else "hybrid_muon_adamw")), device="cpu")
                engine.train_batch(batch(0))
                engine.scaler = torch.amp.GradScaler("cpu", init_scale=1., enabled=True)
                before = engine.state_dict()
                for parameter in model.parameters():
                    parameter.register_hook(lambda gradient: torch.full_like(gradient, 1e20))
                with self.assertRaises(FloatingPointError):
                    engine.train_batch(batch(1))
                after = engine.state_dict()
                for key in ("model", "optimizer", "scheduler", "global_step", "consumed_tokens"):
                    same_tree(before[key], after[key])
                self.assertEqual(engine.overflow_events, 2)
                self.assertEqual(engine.scaler.get_scale(), .25)

    def test_retry_then_success_commits_controller_and_targets_once(self):
        config = replace(_tiny_moe_config(), load_balancing="loss_free_sign", balancing_step_size=.001)
        engine = MoETrainerEngine(MoESmallLLM(config), TrainerConfig(
            optimizer="hybrid_muon_adamw", precision="fp32", microbatch_size=2,
            schedule="wsd", warmup_tokens=60, stable_tokens=60, decay_tokens=60), device="cpu")
        engine.scaler = torch.amp.GradScaler("cpu", init_scale=1., enabled=True)
        calls = 0
        def overflow_once(gradient):
            nonlocal calls
            calls += 1
            return torch.full_like(gradient, float("inf")) if calls == 1 else gradient
        handle = next(engine.model.parameters()).register_hook(overflow_once)
        try:
            result = engine.train_batch(batch(0))
        finally:
            handle.remove()
        self.assertEqual(calls, 2)
        self.assertEqual(engine.overflow_events, 1)
        self.assertEqual(engine.global_step, 1)
        self.assertEqual(engine.consumed_tokens, 6)
        self.assertEqual(result.step, 1)
        for block in engine.model.blocks:
            router = block.ffn.router
            self.assertEqual(int(router.token_exposure.sum()), 6)
            self.assertTrue(torch.equal(router.expert_updates, router.token_exposure.gt(0).long()))
            expected = .001 * torch.sign(router.token_exposure.float().mean() - router.token_exposure)
            torch.testing.assert_close(router.selection_bias, expected)

    def test_score_bias_changes_selection_but_not_gate(self):
        router = SwitchTop1Router(2, 8, init_std=.01, balancing_step_size=.001)
        with torch.no_grad():
            router.projection.weight[:, 0] = torch.tensor(
                [.6, .4, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]
            ).log()
            router.selection_bias[:2] = torch.tensor([-.125, .125])
        result = router(torch.tensor([[1., 0.]]))
        self.assertEqual(result.expert_indices.item(), 1)
        self.assertAlmostEqual(result.selected_probabilities.item(), .4, places=6)
        result.selected_probabilities.sum().backward()
        self.assertGreater(router.projection.weight.grad.abs().sum().item(), 0)
        # Selecting one ORIGINAL softmax probability still differentiates the
        # denominator through every logit; top-1 must not renormalize the gate to 1.
        self.assertTrue(torch.all(router.projection.weight.grad[:, 0] != 0))
        self.assertLess(router.projection.weight.grad[0, 0].item(), 0)
        self.assertGreater(router.projection.weight.grad[1, 0].item(), 0)
        self.assertIsNone(router.selection_bias.grad)

    def test_forward_and_eval_do_not_advance_controller(self):
        router = SwitchTop1Router(4, 8, init_std=.01, balancing_step_size=.001)
        before = {k: v.clone() for k, v in router.named_buffers()}
        for training in (True, False):
            router.train(training)
            router(torch.randn(3, 4))
        for name, value in router.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)
        counts = torch.tensor([4, 0, 0, 0, 1, 1, 1, 1])
        router.commit_load(counts)
        torch.testing.assert_close(
            router.selection_bias,
            torch.tensor([-.001, .001, .001, .001, 0., 0., 0., 0.]),
        )
        self.assertTrue(torch.equal(router.token_exposure, counts))
        self.assertTrue(torch.equal(router.expert_updates, counts.gt(0).long()))

    def test_gamma_zero_matches_outputs_and_gradients(self):
        config = _tiny_moe_config()
        torch.manual_seed(19)
        baseline = DroplessTop1MoE(config)
        candidate = DroplessTop1MoE(replace(
            config, load_balancing="loss_free_sign", balancing_step_size=0.
        ))
        candidate.load_state_dict(baseline.state_dict())
        inputs = torch.randn(2, 3, config.d_model)
        outputs = []
        for model in (baseline, candidate):
            value, z_loss, _ = model(inputs)
            (value.square().mean() + 1e-4 * z_loss).backward()
            outputs.append(value)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for first, second in zip(baseline.parameters(), candidate.parameters()):
            if first.grad is None:
                self.assertIsNone(second.grad)
            else:
                torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_controller_and_exposure_are_checkpoint_state(self):
        config = replace(_tiny_moe_config(), load_balancing="loss_free_sign",
                         balancing_step_size=.001)
        model = MoESmallLLM(config)
        for block in model.blocks:
            block.ffn.router.commit_load(torch.tensor([8, 0, 0, 0, 0, 0, 0, 0]))
        restored = MoESmallLLM(config)
        restored.load_state_dict(model.state_dict())
        for first, second in zip(model.buffers(), restored.buffers()):
            self.assertTrue(torch.equal(first, second))

    def test_config_rejects_misreported_identity(self):
        config = _tiny_moe_config()
        for key, value in [
            ("version", 999), ("router_scoring", "sigmoid"),
            ("router_combine", "uniform"), ("dispatch", "dense_all_experts"),
            ("expert_type", "other"), ("load_balancing", "unknown"),
            ("balancing_step_size", .001), ("balancing_step_size", float("nan")),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(config, **{key: value})

    def test_mixed_precision_expert_output_combines_into_float_residual(self):
        # Force the dtype boundary independently of CPU autocast's index_copy promotion.
        config = _tiny_moe_config()
        moe = DroplessTop1MoE(config)
        for expert in moe.experts:
            expert.register_forward_hook(lambda _m, _x, output: output.to(torch.bfloat16))
        inputs = torch.randn(2, 3, config.d_model, requires_grad=True)
        output, z_loss, _ = moe(inputs)
        self.assertEqual(output.dtype, torch.float32)
        (output.square().mean() + 1e-4 * z_loss).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a GPU")
    def test_cuda_precision_forward_backward_step(self):
        from tests.trainer_fixtures import batch
        for precision in ("fp16", "bf16"):
            with self.subTest(precision=precision):
                config = replace(MoEModelConfig.smoke(gdn_chunk_size=32), load_balancing="loss_free_sign",
                                 balancing_step_size=.001)
                engine = MoETrainerEngine(MoESmallLLM(config), TrainerConfig(
                    optimizer="hybrid_muon_adamw", precision=precision, microbatch_size=1), device="cuda")
                from trainer import TokenBatch
                inputs = torch.randint(0, 16, (2, 65))
                labels = torch.randint(0, 16, (2, 65))
                result = engine.train_batch(TokenBatch(0, "train", inputs, labels, 2, 130))
                self.assertEqual(result.step, 1)
                for block in engine.model.blocks:
                    self.assertEqual(int(block.ffn.router.token_exposure.sum()),
                                     result.target_tokens)


if __name__ == "__main__":
    unittest.main()
