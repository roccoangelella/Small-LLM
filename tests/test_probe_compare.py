from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import torch

from trainer.probe_compare import compare_probes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route(indices: list[int], *, shift: float = 0.0) -> dict[str, torch.Tensor]:
    logits = torch.tensor(
        [[float(index) + shift + (expert == index) for expert in range(8)] for index in indices],
        dtype=torch.float32,
    )
    probabilities = torch.softmax(logits, dim=-1)
    selected = probabilities.gather(1, torch.tensor(indices).unsqueeze(1)).squeeze(1)
    return {
        "logits": logits,
        "expert_indices": torch.tensor(indices, dtype=torch.int64),
        "selected_probabilities": selected,
        "selection_bias": torch.zeros(8, dtype=torch.float32),
    }


def _probe(
    root: Path,
    name: str,
    *,
    step: int,
    tokens: int,
    ce: torch.Tensor,
    routing: dict[str, list[dict[str, torch.Tensor]]],
    probe_sha: str = "probe-sha",
) -> Path:
    payload = {
        "version": 1,
        "checkpoint_id": f"step-{step:08d}",
        "step": step,
        "consumed_tokens": tokens,
        "probe_sha256": probe_sha,
        "checkpoint_manifest_sha256": "not-bound-in-this-test",
        "per_token_ce_fp32": ce.float(),
        "target_mask": torch.tensor(
            [[True, False, True], [False, True, True]], dtype=torch.bool
        ),
        "routing": routing,
    }
    path = root / name
    torch.save(payload, path)
    return path


def _checkpoint(root: Path, *, step: int, tokens: int, model: dict[str, torch.Tensor]) -> Path:
    checkpoint = root / f"step-{step:08d}"
    checkpoint.mkdir()
    state_path = checkpoint / "trainer_state.pkl"
    with state_path.open("wb") as handle:
        pickle.dump(
            {
                "version": 1,
                "global_step": step,
                "consumed_tokens": tokens,
                "model": model,
            },
            handle,
        )
    metadata = {
        "version": 1,
        "checkpoint_id": checkpoint.name,
        "configuration_hash": "a" * 64,
        "source_hash": "b" * 64,
        "schema_hash": "c" * 64,
        "optimizer_step_complete": True,
        "pipeline_state": {},
    }
    metadata_path = checkpoint / "checkpoint.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint / "local_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"name": "trainer_state.pkl", "sha256": _sha256(state_path)},
                    {"name": "checkpoint.json", "sha256": _sha256(metadata_path)},
                ]
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


class ProbeCompareTests(unittest.TestCase):
    def test_ce_mask_routing_churn_js_and_checkpoint_interval_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before_ce = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            after_ce = torch.tensor([[1.5, 1.0, 3.5], [4.0, 5.25, 5.0]])
            before_routing = {
                "blocks.0.ffn.router": [_route([0, 1, 2]), _route([3, 4, 5])]
            }
            after_routing = {
                "blocks.0.ffn.router": [_route([1, 1, 2], shift=0.2), _route([3, 6, 5], shift=0.2)]
            }
            before_probe = _probe(
                root,
                "before.pt",
                step=0,
                tokens=0,
                ce=before_ce,
                routing=before_routing,
            )
            after_probe = _probe(
                root,
                "after.pt",
                step=2,
                tokens=12,
                ce=after_ce,
                routing=after_routing,
            )

            before_model = {
                "blocks.0.ffn.experts.0.gate.weight": torch.ones(2, 2),
                "blocks.0.ffn.experts.0.up.weight": torch.ones(2, 2),
                "blocks.0.ffn.router.projection.weight": torch.ones(8, 2),
                "unrelated.buffer": torch.zeros(1),
            }
            after_model = {name: value.clone() for name, value in before_model.items()}
            after_model["blocks.0.ffn.experts.0.gate.weight"][0, 0] += 1
            after_model["blocks.0.ffn.router.projection.weight"][0, 0] += 0.5
            before_checkpoint = _checkpoint(root, step=0, tokens=0, model=before_model)
            after_checkpoint = _checkpoint(root, step=2, tokens=12, model=after_model)

            # Bind each probe to the exact checkpoint.json hash, as observation.py does.
            for probe_path, checkpoint in (
                (before_probe, before_checkpoint),
                (after_probe, after_checkpoint),
            ):
                payload = torch.load(probe_path, map_location="cpu", weights_only=True)
                payload["checkpoint_manifest_sha256"] = _sha256(checkpoint / "checkpoint.json")
                torch.save(payload, probe_path)

            result = compare_probes(
                before_probe,
                after_probe,
                checkpoint_before=before_checkpoint,
                checkpoint_after=after_checkpoint,
            )
            self.assertEqual(result["ce_delta_after_minus_before"]["count"], 4)
            self.assertAlmostEqual(
                result["ce_delta_after_minus_before"]["mean"], 0.0625, places=6
            )
            routing = result["routing"]["aggregate"]
            self.assertEqual(routing["assignment_churn_count"], 2)
            self.assertEqual(routing["token_count"], 6)
            self.assertEqual(routing["transition_counts_8x8"][0][1], 1)
            self.assertEqual(routing["transition_counts_8x8"][4][6], 1)
            self.assertGreater(routing["router_softmax_js"]["mean"], 0.0)
            groups = result["checkpoint_weight_delta"]["groups"]
            self.assertEqual({group["group"] for group in groups}, {"expert_00", "router"})
            expert = next(group for group in groups if group["group"] == "expert_00")
            self.assertGreater(expert["weight_delta_l2"], 0.0)
            self.assertGreater(expert["weight_delta_over_before_l2"], 0.0)

    def test_dense_empty_routing_is_valid_and_target_mask_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ce = torch.ones(2, 3)
            before = _probe(root, "before.pt", step=0, tokens=0, ce=ce, routing={})
            after = _probe(root, "after.pt", step=1, tokens=6, ce=ce + 1, routing={})
            result = compare_probes(before, after)
            self.assertEqual(result["routing"]["aggregate"]["token_count"], 0)
            self.assertIsNone(result["routing"]["aggregate"]["router_softmax_js"])

            payload = torch.load(after, map_location="cpu", weights_only=True)
            payload["target_mask"] = torch.zeros(2, 3, dtype=torch.bool)
            torch.save(payload, after)
            with self.assertRaisesRegex(ValueError, "target_mask"):
                compare_probes(before, after)


if __name__ == "__main__":
    unittest.main()
