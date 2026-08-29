from __future__ import annotations

import unittest

from trainer.deep_decay_provider_migration import (
    execution_rewrite_needed,
    rewrite_execution_state,
    validate_execution_state,
    validate_target_execution_state,
)


def _state(microbatch: int, rng_count: int) -> dict[str, object]:
    config = {
        "microbatch_size": microbatch,
        "schedule": "wsqd",
        "learning_rate": 3e-4,
    }
    return {
        "config": config,
        "scheduler": {"config": dict(config), "committed_tokens": 123},
        "cuda_rng_states": [bytearray([index, 7, 9]) for index in range(rng_count)],
        "model": {"weight": "sentinel"},
        "optimizer": {"step": 12},
        "scaler": {"scale": 65536},
        "python_rng_state": (1, 2, 3),
        "torch_rng_state": bytearray([4, 5]),
    }


class DeepDecayProviderMigrationTests(unittest.TestCase):
    def test_modal_to_kaggle_duplicates_rank_zero_rng(self) -> None:
        state = _state(16, 1)
        patched, metadata = rewrite_execution_state(state, target_microbatch=2)

        self.assertEqual(patched["config"]["microbatch_size"], 2)  # type: ignore[index]
        self.assertEqual(patched["scheduler"]["config"]["microbatch_size"], 2)  # type: ignore[index]
        rng = patched["cuda_rng_states"]
        self.assertEqual(len(rng), 2)  # type: ignore[arg-type]
        self.assertEqual(rng[0], rng[1])  # type: ignore[index]
        self.assertIsNot(rng[0], rng[1])  # type: ignore[index]
        self.assertEqual(metadata["cuda_rng_policy"], "duplicate_rank0")
        validate_target_execution_state(patched, target_microbatch=2)

    def test_kaggle_to_single_gpu_projects_rank_zero(self) -> None:
        state = _state(2, 2)
        state["cuda_rng_states"] = [bytearray([1]), bytearray([2])]
        patched, metadata = rewrite_execution_state(state, target_microbatch=4)

        self.assertEqual(patched["cuda_rng_states"], [bytearray([1])])
        self.assertEqual(metadata["cuda_rng_policy"], "project_rank0")
        validate_target_execution_state(patched, target_microbatch=4)

    def test_legacy_one_rng_kaggle_state_is_canonicalized(self) -> None:
        state = _state(2, 1)
        self.assertTrue(execution_rewrite_needed(state, target_microbatch=2))
        patched, _ = rewrite_execution_state(state, target_microbatch=2)
        self.assertEqual(len(patched["cuda_rng_states"]), 2)  # type: ignore[arg-type]
        validate_target_execution_state(patched, target_microbatch=2)

    def test_canonical_target_does_not_need_rewrite(self) -> None:
        state = _state(16, 1)
        self.assertFalse(execution_rewrite_needed(state, target_microbatch=16))
        self.assertEqual(validate_execution_state(state), (16, 1))

    def test_unauthorized_microbatch_fails_closed(self) -> None:
        state = _state(8, 1)
        with self.assertRaisesRegex(RuntimeError, "not an authorized migration source"):
            validate_execution_state(state)

    def test_scheduler_config_drift_fails_closed(self) -> None:
        state = _state(4, 1)
        state["scheduler"]["config"]["microbatch_size"] = 16  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "scheduler config disagrees"):
            validate_execution_state(state)


if __name__ == "__main__":
    unittest.main()
