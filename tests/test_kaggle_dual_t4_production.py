"""CPU-only contract tests for standard Kaggle exact-batch dual-T4 training."""
from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

SPEC = importlib.util.spec_from_file_location(
    "kaggle_dual_t4_runtime_test",
    KAGGLE / "dual_t4_runtime.py",
)
assert SPEC is not None and SPEC.loader is not None
runtime_adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_adapter)

TRAIN_SPEC = importlib.util.spec_from_file_location(
    "kaggle_dual_t4_train_test",
    KAGGLE / "dual_t4_train.py",
)
assert TRAIN_SPEC is not None and TRAIN_SPEC.loader is not None
production = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(production)


class KaggleDualT4ProductionTests(unittest.TestCase):
    def test_online_command_uses_two_process_torchrun_and_qualified_runtime(self) -> None:
        command = [
            "/usr/bin/uv",
            "run",
            "--python",
            "3.13",
            "--extra",
            "model",
            "python",
            "-m",
            "trainer",
            "--dataset-dir",
            "/data/run",
            "--wandb-tags",
            "20m",
            "2b-tokens",
            "t4",
            "--wandb-resume",
            "allow",
        ]
        worktree = Path("/tmp/pinned-worktree")
        rewritten = runtime_adapter.distributed_trainer_command(command, worktree=worktree)
        self.assertIn("torch.distributed.run", rewritten)
        self.assertIn("--nproc-per-node=2", rewritten)
        self.assertIn(str(runtime_adapter.DDP_ENTRYPOINT), rewritten)
        self.assertIn(str(worktree), rewritten)
        self.assertIn("torch==2.10.0", rewritten)
        self.assertIn("triton==3.6.0", rewritten)
        self.assertIn("fla-core==0.5.2", rewritten)
        self.assertIn("huggingface_hub==1.5.0", rewritten)
        self.assertIn("https://download.pytorch.org/whl/cu128", rewritten)
        self.assertIn("dual-t4-ddp", rewritten)
        self.assertNotIn("trainer", rewritten[rewritten.index("python") : rewritten.index("--dataset-dir")])
        self.assertEqual(rewritten[rewritten.index("--dataset-dir") + 1], "/data/run")

    def test_runtime_adapter_leaves_offline_probe_single_process(self) -> None:
        module = SimpleNamespace(
            WORKTREE=Path("/tmp/worktree"),
            trainer_command=lambda *args, **kwargs: ["uv", "run", "python", "-m", "trainer"],
        )
        fake_runtime = SimpleNamespace(
            TRAINING_ENGINE=Path("/tmp/engine.py"),
            _load=lambda path, name: module,
        )
        runtime_adapter.install(fake_runtime)
        loaded = fake_runtime._load(Path("/tmp/engine.py"), "engine")
        offline = loaded.trainer_command(online=False)
        self.assertEqual(offline, ["uv", "run", "python", "-m", "trainer"])
        online = loaded.trainer_command(online=True)
        self.assertIn("torch.distributed.run", online)

    def test_qualification_autotune_policy_is_reused(self) -> None:
        self.assertEqual(production.AUTOTUNE_CONFIG_CAP, 6)
        self.assertEqual(
            production._representative_config_indices(36, 6),
            [0, 7, 14, 21, 28, 35],
        )

    def test_prewarm_can_use_a_smaller_sft_execution_microbatch(self) -> None:
        parameter = inspect.signature(production._prewarm_raw_model).parameters[
            "microbatch_size"
        ]
        self.assertEqual(parameter.default, production.MICROBATCH_SIZE)

        source = inspect.getsource(production._prewarm_raw_model)
        self.assertIn("(microbatch_size, CONTEXT_LENGTH)", source)

    def test_overflow_collectives_precede_any_optimizer_step(self) -> None:
        source = (KAGGLE / "dual_t4_train.py").read_text(encoding="utf-8")
        synchronize = source.index("dist.all_reduce(flags, op=dist.ReduceOp.MAX)")
        unanimity = source.index("dist.all_reduce(unanimous, op=dist.ReduceOp.MIN)")
        optimizer_step = source.index("engine.scaler.step(engine.optimizer)")
        self.assertLess(synchronize, unanimity)
        self.assertLess(unanimity, optimizer_step)
        self.assertIn("asymmetric GradScaler found_inf across DDP ranks", source)
        self.assertIn("GradScaler skips the underlying", source)

    def test_checkpoint_and_validation_use_raw_model_adapter(self) -> None:
        sentinel_raw = object()
        sentinel_wrapper = object()
        engine = SimpleNamespace(model=sentinel_wrapper, _small_llm_raw_model=sentinel_raw)

        def inspect(current: object) -> object:
            self.assertIs(current.model, sentinel_raw)
            return "ok"

        self.assertEqual(production._with_raw_model(engine, inspect), "ok")
        self.assertIs(engine.model, sentinel_wrapper)

    def test_modal_launch_does_not_enable_kaggle_ddp(self) -> None:
        modal_launch = (ROOT / "modal" / "launch.py").read_text(encoding="utf-8")
        self.assertNotIn("dual_t4_runtime", modal_launch)
        self.assertNotIn("dual_t4_train", modal_launch)
        self.assertNotIn("torch.distributed.run", modal_launch)


if __name__ == "__main__":
    unittest.main()
