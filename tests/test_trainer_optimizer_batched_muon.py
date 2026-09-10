"""CPU contracts: batched-by-shape Muon equals the per-matrix reference path."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from trainer.config import TrainerConfig
from trainer.optimizer import (
    HybridMuonAdamW,
    _newton_schulz_orthogonalize,
    _newton_schulz_orthogonalize_batched,
    build_hybrid_muon_adamw,
)
from trainer.optimizer_telemetry import InstrumentedHybridMuonAdamW


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)


class _Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = _FFN(8, 12)  # names end with ".ffn.{gate,up,down}.weight" -> Muon


class _Block(nn.Module):
    def __init__(self, experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(_Expert() for _ in range(experts))
        self.mixer = nn.Module()
        self.mixer.q_proj = nn.Linear(8, 8, bias=False)
        self.mixer.out_proj = nn.Linear(8, 8, bias=False)
        self.ffn_norm = nn.LayerNorm(8, elementwise_affine=True, bias=False)


class _Model(nn.Module):
    """Several same-shape matrices (batched buckets) plus singletons."""

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(16, 8)
        self.blocks = nn.ModuleList(_Block(3) for _ in range(2))
        self.final_norm = nn.LayerNorm(8, elementwise_affine=True, bias=False)


def _config(**overrides: object) -> TrainerConfig:
    base = dict(
        optimizer="hybrid_muon_adamw",
        precision="fp32",
        learning_rate=1e-2,
        weight_decay=0.1,
        muon_weight_decay=0.05,
    )
    base.update(overrides)
    return TrainerConfig(**base)  # type: ignore[arg-type]


def _muon_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2 and (".ffn." in name or ".mixer." in name)
    ]


def _seed_gradients(model: nn.Module, seed: int, *, zero_first_expert: bool = False) -> None:
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in model.named_parameters():
        parameter.grad = torch.randn(parameter.shape, generator=generator)
        if zero_first_expert and name.endswith("blocks.0.ffn.experts.0.gate.weight"):
            parameter.grad.zero_()


def _reference_step(optimizer: HybridMuonAdamW) -> None:
    """The original per-matrix Muon path, kept as the numerical oracle."""

    with torch.no_grad():
        for group in optimizer.param_groups:
            role = group.get("optimizer_role")
            if role == "muon":
                for parameter in group["params"]:
                    optimizer._muon_step(parameter, group)
            else:
                for parameter in group["params"]:
                    optimizer._adamw_step(parameter, group)


class TestBatchedNewtonSchulz(unittest.TestCase):
    def test_batched_matches_single_matrix_for_both_orientations(self) -> None:
        for shape in ((5, 9), (9, 5), (6, 6)):
            stack = torch.randn(7, *shape, generator=torch.Generator().manual_seed(3))
            batched = _newton_schulz_orthogonalize_batched(stack, target_rms=0.18)
            for index in range(stack.shape[0]):
                single = _newton_schulz_orthogonalize(stack[index], target_rms=0.18)
                torch.testing.assert_close(batched[index], single, rtol=1e-5, atol=1e-6)

    def test_zero_matrix_inside_a_batch_stays_zero_without_poisoning_neighbours(self) -> None:
        stack = torch.randn(4, 5, 7, generator=torch.Generator().manual_seed(5))
        stack[2].zero_()
        batched = _newton_schulz_orthogonalize_batched(stack, target_rms=0.18)
        self.assertTrue(torch.equal(batched[2], torch.zeros(5, 7)))
        for index in (0, 1, 3):
            single = _newton_schulz_orthogonalize(stack[index], target_rms=0.18)
            torch.testing.assert_close(batched[index], single, rtol=1e-5, atol=1e-6)

    def test_non_finite_slice_is_rejected(self) -> None:
        stack = torch.randn(3, 4, 4)
        stack[1, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            _newton_schulz_orthogonalize_batched(stack, target_rms=0.18)


class TestBatchedMuonStep(unittest.TestCase):
    def _pair(self, seed: int = 11) -> tuple[nn.Module, nn.Module]:
        torch.manual_seed(seed)
        reference = _Model()
        candidate = copy.deepcopy(reference)
        return reference, candidate

    def test_grouped_step_matches_per_matrix_reference_over_several_steps(self) -> None:
        reference, candidate = self._pair()
        ref_opt = build_hybrid_muon_adamw(reference, _config())
        new_opt = build_hybrid_muon_adamw(candidate, _config())
        for step in range(3):
            _seed_gradients(reference, 100 + step, zero_first_expert=(step == 1))
            _seed_gradients(candidate, 100 + step, zero_first_expert=(step == 1))
            _reference_step(ref_opt)
            new_opt.step()
            for (name, expected), (_, actual) in zip(
                reference.named_parameters(), candidate.named_parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)
                # CPU kernels are deterministic: the grouped path reproduces the
                # per-matrix reference bit for bit (GPU batched GEMMs may round
                # differently, hence the tolerance above as the portable contract).
                self.assertTrue(torch.equal(actual, expected), name)
            for ref_param, new_param in zip(_muon_parameters(reference), _muon_parameters(candidate)):
                torch.testing.assert_close(
                    new_opt.state[new_param]["momentum_buffer"],
                    ref_opt.state[ref_param]["momentum_buffer"],
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_parameters_without_gradient_are_skipped(self) -> None:
        reference, candidate = self._pair()
        ref_opt = build_hybrid_muon_adamw(reference, _config())
        new_opt = build_hybrid_muon_adamw(candidate, _config())
        _seed_gradients(reference, 7)
        _seed_gradients(candidate, 7)
        reference.blocks[1].experts[2].ffn.down.weight.grad = None
        candidate.blocks[1].experts[2].ffn.down.weight.grad = None
        before = candidate.blocks[1].experts[2].ffn.down.weight.detach().clone()
        _reference_step(ref_opt)
        new_opt.step()
        self.assertTrue(torch.equal(candidate.blocks[1].experts[2].ffn.down.weight, before))
        self.assertNotIn("momentum_buffer", new_opt.state[candidate.blocks[1].experts[2].ffn.down.weight])
        for (name, expected), (_, actual) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)

    def test_non_finite_gradient_fails_before_mutating_its_bucket(self) -> None:
        _, candidate = self._pair()
        new_opt = build_hybrid_muon_adamw(candidate, _config())
        _seed_gradients(candidate, 9)
        target = candidate.blocks[0].experts[1].ffn.up.weight
        target.grad[0, 0] = float("inf")
        bucket = [
            parameter
            for parameter in _muon_parameters(candidate)
            if tuple(parameter.shape) == tuple(target.shape)
        ]
        snapshot = [parameter.detach().clone() for parameter in bucket]
        with self.assertRaises(FloatingPointError):
            new_opt.step()
        for parameter, before in zip(bucket, snapshot):
            self.assertTrue(torch.equal(parameter, before))
            self.assertNotIn("momentum_buffer", new_opt.state[parameter])

    def test_instrumented_optimizer_records_every_muon_matrix_with_same_direction_rms(self) -> None:
        reference, candidate = self._pair()
        ref_opt = build_hybrid_muon_adamw(reference, _config())
        new_opt = InstrumentedHybridMuonAdamW(new_opt_classified(candidate), _config())
        _seed_gradients(reference, 21)
        _seed_gradients(candidate, 21)
        _reference_step(ref_opt)
        new_opt.step()
        statistics = new_opt.step_statistics()
        muon_stats = statistics["muon"]
        self.assertEqual(muon_stats["parameter_tensors"], len(_muon_parameters(candidate)))
        per_matrix = muon_stats["matrix_optimizer_direction_rms"]
        self.assertEqual(len(per_matrix), len(_muon_parameters(candidate)))
        for value in per_matrix.values():
            self.assertAlmostEqual(float(value), 0.18, places=5)
        for (name, expected), (_, actual) in zip(
            reference.named_parameters(), candidate.named_parameters()
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)


def new_opt_classified(model: nn.Module):
    from trainer.optimizer import _classify_parameters

    return _classify_parameters(model)


if __name__ == "__main__":
    unittest.main()
