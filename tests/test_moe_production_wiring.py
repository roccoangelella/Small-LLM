"""Regression guards for the accepted 64E/Top-2 production entry path."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from MOE_model.__main__ import parse_args
from MOE_model.router import SwitchTopKRouter
from MOE_model.setup import _model_config_from_args
from MOE_model.step import _finalize_telemetry


class TestAcceptedCLIWiring(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "model_size": "accepted",
            "architecture": "gdn2_hybrid",
            "gdn_chunk_size": 32,
            "load_balancing": None,
            "balancing_step_size": 0.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_accepted_model_is_reachable_and_keeps_quantile_intrinsic(self) -> None:
        config = _model_config_from_args(self._args())
        self.assertEqual((config.num_experts, config.top_k, config.expert_d_ff), (64, 2, 352))
        self.assertEqual(config.load_balancing, "quantile")
        self.assertEqual((config.semantic_vocab_size, config.padded_vocab_size), (8_000, 8_192))

    def test_generic_balancing_override_cannot_downgrade_production(self) -> None:
        for controller in ("none", "loss_free_sign"):
            with self.subTest(controller=controller), self.assertRaises(ValueError):
                _model_config_from_args(self._args(load_balancing=controller))

    def test_cli_accepts_the_production_identity_and_quantile_confirmation(self) -> None:
        args = parse_args([
            "--dataset-dir", ".", "--checkpoint-dir", ".", "--steps", "1",
            "--model-size", "accepted", "--device", "cpu", "--precision", "fp32",
            "--gdn-chunk-size", "32", "--load-balancing", "quantile",
        ])
        self.assertEqual(args.model_size, "accepted")
        self.assertEqual(args.load_balancing, "quantile")

    def test_cli_rejects_downgrading_accepted_balancing(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([
                "--dataset-dir", ".", "--checkpoint-dir", ".", "--steps", "1",
                "--model-size", "accepted", "--device", "cpu", "--precision", "fp32",
                "--gdn-chunk-size", "32", "--load-balancing", "none",
            ])


class TestQuantileTransactionalState(unittest.TestCase):
    def test_abandoned_attempt_does_not_change_bias_or_contaminate_retry(self) -> None:
        torch.manual_seed(71)
        router = SwitchTopKRouter(
            8, 4, init_std=0.02, top_k=2, scoring="sqrt_softplus", balancing="quantile"
        )
        router.train()

        first = torch.randn(64, 8)
        router.begin_step(first.shape[0])
        first_route = router(first)
        router.commit_load(first_route.expert_counts.long())
        committed = router.selection_bias.clone()

        # Simulate an optimizer attempt that overflows: observe scores but never commit.
        abandoned = torch.randn(64, 8) + 10.0
        router.begin_step(abandoned.shape[0])
        router(abandoned)
        torch.testing.assert_close(router.selection_bias, committed, rtol=0, atol=0)

        # Retrying opens a fresh transient frontier. The resulting bias must equal a
        # clean controller that saw only the retry scores, not the abandoned attempt.
        retry = torch.randn(64, 8)
        router.begin_step(retry.shape[0])
        retry_route = router(retry)
        router.commit_load(retry_route.expert_counts.long())
        actual = router.selection_bias.clone()

        clean = SwitchTopKRouter(
            8, 4, init_std=0.02, top_k=2, scoring="sqrt_softplus", balancing="quantile"
        )
        clean.load_state_dict(router.state_dict())
        with torch.no_grad():
            clean.selection_bias.copy_(committed)
        clean.begin_step(retry.shape[0])
        clean_route = clean(retry)
        clean.commit_load(clean_route.expert_counts.long())
        torch.testing.assert_close(actual, clean.selection_bias, rtol=0, atol=0)


class _Accounting:
    def as_dict(self):
        return {"total": 1}


class TestProductionTelemetry(unittest.TestCase):
    def test_top_k_comes_from_model_identity(self) -> None:
        routers = [
            SimpleNamespace(
                token_exposure=torch.zeros(4, dtype=torch.long),
                expert_updates=torch.zeros(4, dtype=torch.long),
                selection_bias=torch.zeros(4),
            )
            for _ in range(2)
        ]
        engine = SimpleNamespace(
            device=torch.device("cpu"),
            model=SimpleNamespace(
                config=SimpleNamespace(num_experts=4, top_k=2),
                blocks=[SimpleNamespace(ffn=SimpleNamespace(router=router)) for router in routers],
            ),
            parameter_accounting=_Accounting(),
        )
        accumulator = [
            {
                "counts": torch.tensor([2, 2, 2, 2], dtype=torch.long),
                "selected_probability_sum": torch.tensor(4.0),
                "entropy_sum": torch.tensor(3.0),
                "token_count": 4,
            }
            for _ in range(2)
        ]
        payload = _finalize_telemetry(engine, accumulator)
        self.assertEqual(payload["routing"]["top_k"], 2)
        self.assertEqual(payload["routing"]["num_experts"], 4)


if __name__ == "__main__":
    unittest.main()
