"""Execution-only MoE DDP must preserve the frozen global router contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest import TestCase, mock

import torch
from MOE_model.router import SwitchTopKRouter
from dataset.incremental_frontier import FrontierShard

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("small_llm_moe_ddp", ROOT / "kaggle/moe_train_dual_t4.py")
assert SPEC is not None and SPEC.loader is not None
moe_ddp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(moe_ddp)


class KaggleMoEDDPTests(TestCase):
    def test_wall_drain_from_either_rank_reaches_both(self) -> None:
        with mock.patch.object(torch.distributed, "all_reduce", side_effect=lambda flag, op: flag.fill_(1)):
            self.assertTrue(moe_ddp.any_rank_drain(False, torch.device("cpu")))
            self.assertTrue(moe_ddp.any_rank_drain(True, torch.device("cpu")))

    def test_global_quantile_merges_both_ranks_and_commits_counts(self) -> None:
        router = SwitchTopKRouter(4, 4, init_std=0.02, top_k=2,
                                   scoring="sqrt_softplus", balancing="quantile")
        router.begin_step(8)  # global target is fourth-largest score / expert
        local = torch.tensor([[9., 5., 4., 1.], [8., 6., 2., 0.],
                              [7., 4., 3., 2.], [5., 4., 2., 1.]])
        peer = torch.tensor([[10., 8., 7., 3.], [9., 7., 5., 1.],
                             [10., 6., 5., 4.], [8., 7., 6., 3.]])
        router._score_frontier = local.clone()
        counts = torch.tensor([2, 1, 3, 2], dtype=torch.long)
        item: dict[str, object] = {
            "counts": counts, "selected_probability_sum": torch.tensor(4.),
            "entropy_sum": torch.tensor(2.), "token_count": 4,
        }
        def reduce(tensor: torch.Tensor, op: object) -> None:
            if tensor.dtype == torch.long:
                tensor.add_(torch.tensor([1, 3, 0, 4]))
            else:
                tensor.mul_(2)
        def gather(targets: list[torch.Tensor], own: torch.Tensor) -> None:
            targets[0].copy_(own)
            targets[1].copy_(peer)
        with mock.patch("torch.distributed.all_reduce", side_effect=reduce), \
             mock.patch("torch.distributed.all_gather", side_effect=gather):
            moe_ddp.merge_router_frontiers([router], [item])
        thresholds = torch.cat((local, peer), dim=1).topk(4, dim=1).values[:, 3]
        torch.testing.assert_close(router.selection_bias, thresholds.mean() - thresholds)
        torch.testing.assert_close(router.token_exposure, torch.tensor([3, 4, 3, 6]))
        torch.testing.assert_close(item["counts"], torch.tensor([3, 4, 3, 6]))
        self.assertEqual(item["token_count"], 8)
        self.assertIsNone(router._score_frontier)

    def test_follower_reads_verified_lead_shard_but_never_evicted_it(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blob = b"two rank immutable shard"
            shard = FrontierShard("train/train-000000.bin", "train", len(blob),
                                   hashlib.sha256(blob).hexdigest(), 0, 0, 64)
            (root / "train").mkdir()
            path = root / shard.filename
            path.write_bytes(blob)
            (root / "run_contract.json").write_text(json.dumps({
                "run_id": "v2", "contract_sha256": "same", "planned_train_blocks": 1,
            }))
            (root / "shard_frontier.json").write_text(json.dumps({
                "run_id": "v2", "contract_sha256": "same", "producer_complete": True,
                "ready_train_shards": [shard.as_dict()],
            }))
            follower = moe_ddp.FollowerCache(root, timeout=0.1)
            follower.ensure_block(0)
            follower.acknowledge(0)
            self.assertEqual(path.read_bytes(), blob)
            self.assertEqual(follower.planned_block_count, 1)

    def test_follower_fails_closed_on_different_contract(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_contract.json").write_text(json.dumps({
                "run_id": "v2", "contract_sha256": "good", "planned_train_blocks": 1,
            }))
            (root / "shard_frontier.json").write_text(json.dumps({
                "run_id": "v2", "contract_sha256": "bad", "producer_complete": True,
                "ready_train_shards": [],
            }))
            with self.assertRaisesRegex(RuntimeError, "matching"):
                moe_ddp.FollowerCache(root)
