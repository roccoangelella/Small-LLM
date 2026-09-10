"""CPU contracts: the sorted single-sync MoE dispatch reproduces the previous
per-expert ``nonzero`` dispatch bit for bit, forward and backward."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from model.config import ModelConfig
from MOE_model.config import MoEModelConfig
from MOE_model.model import DroplessTop1MoE


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
    return MoEModelConfig(dense=dense, expert_d_ff=96, moe_layer_indices=tuple(range(4)))


def _reference_forward(moe: DroplessTop1MoE, x: Tensor) -> tuple[Tensor, Tensor]:
    """The previous dispatch: per-expert ``nonzero`` + out-of-place ``index_copy``."""

    shape = x.shape
    flat = x.reshape(-1, moe.d_model)
    route = moe.router(flat)
    combined = torch.zeros_like(flat)
    for expert_id, expert in enumerate(moe.experts):
        token_indices = torch.nonzero(route.expert_indices == expert_id, as_tuple=False).flatten()
        if token_indices.numel() == 0:
            continue
        expert_outputs = expert(flat.index_select(0, token_indices))
        gates = route.selected_probabilities.index_select(0, token_indices).to(
            dtype=expert_outputs.dtype
        ).unsqueeze(-1)
        combined = combined.index_copy(
            0, token_indices, (expert_outputs * gates).to(dtype=combined.dtype)
        )
    return combined.reshape(shape), route.z_loss


class SortedDispatchTests(unittest.TestCase):
    def _compare(self, seed: int, batch: int, sequence: int, *, collapse: bool = False) -> None:
        torch.manual_seed(seed)
        config = _tiny_moe_config()
        moe = DroplessTop1MoE(config)
        if collapse:
            with torch.no_grad():
                moe.router.projection.weight.zero_()
                moe.router.projection.weight[3].fill_(0.1)
        reference = copy.deepcopy(moe)
        x = torch.randn(batch, sequence, config.d_model)
        x_new = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        y_new, z_new, telemetry = moe(x_new)
        y_ref, z_ref = _reference_forward(reference, x_ref)
        self.assertTrue(torch.equal(y_new, y_ref))
        self.assertTrue(torch.equal(z_new, z_ref))
        self.assertEqual(int(telemetry.expert_counts.sum()), batch * sequence)
        (y_new.square().mean() + 1e-4 * z_new).backward()
        (y_ref.square().mean() + 1e-4 * z_ref).backward()
        self.assertTrue(torch.equal(x_new.grad, x_ref.grad))
        for (name, p_new), (_, p_ref) in zip(moe.named_parameters(), reference.named_parameters()):
            if p_ref.grad is None:
                self.assertIsNone(p_new.grad, name)
            else:
                self.assertTrue(torch.equal(p_new.grad, p_ref.grad), name)

    def test_matches_previous_dispatch_bitwise_on_random_routing(self) -> None:
        for seed, batch, sequence in ((1, 2, 7), (2, 3, 8), (3, 1, 5)):
            self._compare(seed, batch, sequence)

    def test_matches_previous_dispatch_when_all_tokens_pick_one_expert(self) -> None:
        self._compare(9, 2, 6, collapse=True)

    def test_dispatch_issues_no_nonzero_and_one_boundary_read_per_layer(self) -> None:
        torch.manual_seed(4)
        moe = DroplessTop1MoE(_tiny_moe_config())
        x = torch.randn(2, 8, 64)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            moe(x)
        counts = {event.key: event.count for event in prof.key_averages()}
        self.assertEqual(counts.get("aten::nonzero", 0), 0)
        self.assertEqual(counts.get("aten::index_copy", 0), 0)
        self.assertEqual(counts.get("aten::index_copy_", 0), 1)
        self.assertEqual(counts.get("aten::cumsum", 0), 1)


if __name__ == "__main__":
    unittest.main()
