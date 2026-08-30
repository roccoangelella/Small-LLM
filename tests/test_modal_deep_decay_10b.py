"""Fail-closed CPU contracts for the Modal 100M/10B deep-decay lane."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import unittest

from model.config import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / "modal"


def _load():
    spec = importlib.util.spec_from_file_location(
        "small_llm_modal_deep_decay_test",
        MODAL / "deep_decay_10b_from_15500.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _continuation_state(module, *, step: int = 27_750, microbatch: int = 2):
    config = {
        **module._scientific_config(),
        "microbatch_size": microbatch,
        "checkpoint_every_steps": 250,
        "evaluation_every_steps": 250,
        "log_every_steps": 1,
    }
    committed = step * module.TARGETS_PER_FULL_BLOCK
    return {
        "version": 1,
        "config": config,
        "model_config": asdict(
            ModelConfig.substantive(architecture="gdn2_hybrid", gdn_chunk_size=32)
        ),
        "model": {"weight": object()},
        "optimizer": {"state": object()},
        "scheduler": {
            "config": dict(config),
            "committed_tokens": committed,
            "last_lr": module.expected_lr(committed),
        },
        "scaler": {"scale": 1.0},
        "global_step": step,
        "consumed_tokens": committed,
        "python_rng_state": object(),
        "torch_rng_state": object(),
        "cuda_rng_states": [object(), object()] if microbatch == 2 else [object()],
    }


class ModalDeepDecayTests(unittest.TestCase):
    def test_modal_schedule_namespace_and_h100_slicing_are_frozen(self) -> None:
        module = _load()

        self.assertEqual(module.RUN_ID, "100m-10b-deep-decay-from-step15500")
        self.assertEqual(module.SOURCE_CHECKPOINT_ID, "step-00015500")
        self.assertEqual(module.SOURCE_EXPECTED_TOKENS, 2_031_616_000)
        self.assertEqual(module.SEQUENCES_PER_BLOCK, 64)
        self.assertEqual(module.MICROBATCH_SIZE, 16)
        self.assertEqual(module.SETTLE_END_STEP, 17_789)
        self.assertEqual(module.COOLDOWN_START_STEP, 73_242)
        self.assertEqual(module.FINAL_STEP, 76_294)
        self.assertEqual(module.TOTAL_TARGETS, 10_000_007_168)
        self.assertTrue(math.isclose(module.expected_lr(module.SETTLE_END_TOKENS), 1e-4, rel_tol=1e-12))
        self.assertTrue(math.isclose(module.expected_lr(module.COOLDOWN_START_TOKENS), 1e-5, rel_tol=1e-12))
        self.assertTrue(math.isclose(module.expected_lr(module.TOTAL_TARGETS), 5e-6, rel_tol=1e-12))

    def test_existing_continuation_accepts_only_authorized_slicing_sources(self) -> None:
        module = _load()
        state = _continuation_state(module, microbatch=2)
        allowed = module.PRIOR_CONTINUATION_MICROBATCH_SIZES | {module.MICROBATCH_SIZE}

        self.assertEqual(
            module.validate_state(
                state,
                step=27_750,
                source_checkpoint=False,
                allowed_microbatches=allowed,
            ),
            2,
        )
        state["config"]["microbatch_size"] = 8
        state["scheduler"]["config"]["microbatch_size"] = 8
        with self.assertRaisesRegex(RuntimeError, "not an authorized migration source"):
            module.validate_state(
                state,
                step=27_750,
                source_checkpoint=False,
                allowed_microbatches=allowed,
            )

    def test_existing_continuation_fails_closed_on_schedule_lr_or_model_drift(self) -> None:
        module = _load()
        state = _continuation_state(module)
        state["config"]["base_power"] = 0.5
        state["scheduler"]["config"]["base_power"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "scientific config drifted"):
            module.validate_state(
                state,
                step=27_750,
                source_checkpoint=False,
                allowed_microbatches=frozenset({2}),
            )

        state = _continuation_state(module)
        state["scheduler"]["last_lr"] = 3e-4
        with self.assertRaisesRegex(RuntimeError, "scheduler LR drifted"):
            module.validate_state(
                state,
                step=27_750,
                source_checkpoint=False,
                allowed_microbatches=frozenset({2}),
            )

        state = _continuation_state(module)
        state["model_config"]["n_layers"] = 19
        with self.assertRaisesRegex(RuntimeError, "model config drifted"):
            module.validate_state(
                state,
                step=27_750,
                source_checkpoint=False,
                allowed_microbatches=frozenset({2}),
            )

    def test_provider_rewrite_changes_only_microbatch_for_existing_continuation(self) -> None:
        module = _load()
        state = _continuation_state(module)
        patched = module._patched_state(state, source_checkpoint=False)

        self.assertIs(patched["model"], state["model"])
        self.assertIs(patched["optimizer"], state["optimizer"])
        self.assertIs(patched["scaler"], state["scaler"])
        self.assertIs(patched["python_rng_state"], state["python_rng_state"])
        self.assertIs(patched["torch_rng_state"], state["torch_rng_state"])
        self.assertEqual(len(patched["cuda_rng_states"]), 1)
        self.assertIs(patched["cuda_rng_states"][0], state["cuda_rng_states"][0])
        changed = {
            key
            for key in state["config"]
            if state["config"][key] != patched["config"][key]
        }
        self.assertEqual(changed, {"microbatch_size"})
        self.assertEqual(patched["config"]["microbatch_size"], 16)
        self.assertEqual(patched["scheduler"]["config"], patched["config"])
        self.assertEqual(
            patched["scheduler"]["committed_tokens"], state["scheduler"]["committed_tokens"]
        )
        self.assertEqual(patched["scheduler"]["last_lr"], state["scheduler"]["last_lr"])

    def test_cuda_rng_projection_is_rank0_exact_and_topology_fail_closed(self) -> None:
        module = _load()
        state = _continuation_state(module, microbatch=2)
        module.validate_state(
            state,
            step=27_750,
            source_checkpoint=False,
            allowed_microbatches=frozenset({2}),
        )
        projected = module._patched_state(state, source_checkpoint=False)
        module.validate_state(
            projected,
            step=27_750,
            source_checkpoint=False,
            allowed_microbatches=frozenset({16}),
            expected_cuda_rng_states=1,
        )
        self.assertIs(projected["cuda_rng_states"][0], state["cuda_rng_states"][0])

        state = _continuation_state(module, microbatch=2)
        state["cuda_rng_states"].append(object())
        with self.assertRaisesRegex(RuntimeError, "CUDA RNG topology drifted"):
            module.validate_state(
                state,
                step=27_750,
                source_checkpoint=False,
                allowed_microbatches=frozenset({2}),
            )

    def test_pointer_and_data_cursor_validation_fail_closed(self) -> None:
        module = _load()
        with self.assertRaisesRegex(RuntimeError, "not a JSON object"):
            module._pointer_checkpoint_id([], label=module.RUN_ID)
        with self.assertRaisesRegex(RuntimeError, "invalid checkpoint ID"):
            module._pointer_checkpoint_id(
                {"checkpoint_id": "step-27750"},
                label=module.RUN_ID,
            )

        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "pipeline_state": {
                            "last_consumed_block_id": 27_748,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "data cursor drifted"):
                module._verify_checkpoint_payload(root, checkpoint_id="step-00027750")

    def test_launcher_exposes_deep_decay_and_gates_cpu_before_exact_h100(self) -> None:
        source = (MODAL / "launch.py").read_text(encoding="utf-8")
        main = source[source.index("def main(") :]

        self.assertIn('action: str = "train"', main)
        self.assertIn('action == "deep-decay"', main)
        self.assertIn("deep-decay is frozen to --model 100M --tokens 10B", main)
        self.assertIn("prepare_deep_decay_remote.remote(source_commit)", main)
        self.assertIn("train_deep_decay_remote.spawn(", main)
        self.assertLess(
            main.index("prepare_deep_decay_remote.remote(source_commit)"),
            main.index("train_deep_decay_remote.spawn("),
        )
        decorator_start = source.rindex("@app.function(", 0, source.index("def train_deep_decay_remote"))
        decorator = source[decorator_start : source.index("def train_deep_decay_remote")]
        self.assertIn('gpu="H100!"', decorator)
        self.assertNotIn("retries=", decorator)

    def test_dry_run_describes_newest_continuation_first_and_exact_source_fallback(self) -> None:
        module = _load()
        payload = module.dry_run_payload(250)

        self.assertEqual(payload["execution"], "modal_single_h100_block64")
        self.assertEqual(payload["gpu"], "H100!")
        self.assertEqual(payload["continuation_run_id"], module.RUN_ID)
        self.assertEqual(payload["microbatch_size"], 16)
        self.assertEqual(payload["microbatches_per_update"], 4)
        self.assertEqual(
            payload["resume"],
            "newest_verified_continuation_hf_then_exact_step_00015500_only",
        )
        self.assertEqual(payload["max_steps_this_session"], 250)

    def test_local_best_checkpoint_selects_minimum_validation_loss(self) -> None:
        module = _load()
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            for step_str, loss in [("step-00016000", 3.5), ("step-00016250", 3.2), ("step-00016500", 3.4)]:
                step_dir = checkpoint_dir / step_str
                step_dir.mkdir()
                (step_dir / "checkpoint.json").write_text(
                    json.dumps({"validation_metrics": {"loss": loss}}),
                    encoding="utf-8",
                )
                (step_dir / "local_manifest.json").write_text("{}", encoding="utf-8")
                (step_dir / "trainer_state.pkl").write_bytes(b"state")

            cid, val_loss = module._local_best_checkpoint(checkpoint_dir)
            self.assertEqual(cid, "step-00016250")
            self.assertEqual(val_loss, 3.2)


if __name__ == "__main__":
    unittest.main()

