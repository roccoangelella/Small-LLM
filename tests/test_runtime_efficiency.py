"""CPU contract tests for runtime-only training changes."""
import contextlib
import io
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from trainer.async_publication import AsyncPublication
from trainer.cli import main
from trainer import TrainerConfig, TrainerEngine, TokenBatch
from MOE_model.model import MoESmallLLM
from tests.test_moe_evaluation import _tiny_moe_config
from tests.test_trainer_remote_publication import _Engine, _Session, _Coordinator, _BASE


class RuntimeEfficiencyTests(unittest.TestCase):
    def test_moe_validation_batching_preserves_targets_weights_rng_and_loss(self):
        torch.manual_seed(17)
        config = replace(_tiny_moe_config(), version=3, num_experts=64, top_k=2,
                         router_scoring="sqrt_softplus", router_z_loss_coefficient=0.0,
                         router_combine="normalised_topk_softmax", load_balancing="quantile",
                         dispatch="dropless_padded_batched_gemm")
        model = MoESmallLLM(config)
        from MOE_model.initialization import initialize_moe_model
        initialize_moe_model(model)
        engine = TrainerEngine(model, TrainerConfig(precision="fp32", optimizer="adamw"), device="cpu")
        tokens = torch.randint(0, 32, (5, 9))
        batch = TokenBatch(0, "validation", tokens[:, :-1], tokens[:, 1:], 5, 40)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        single = engine.evaluate([batch], microbatch_size=1)
        batched = engine.evaluate([batch], microbatch_size=4)
        self.assertEqual(single["target_tokens"], batched["target_tokens"])
        self.assertEqual(single["blocks"], batched["blocks"])
        torch.testing.assert_close(single["loss"], batched["loss"], rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(model.training)

    def test_async_failure_surfaces_at_join(self):
        def fail():
            raise RuntimeError("upload failed")
        uploader = AsyncPublication(lambda result: None)
        uploader.submit(fail)
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            uploader.close()

    def test_async_failure_surfaces_on_poll_without_waiting_for_checkpoint(self):
        uploader = AsyncPublication(lambda result: None)
        def fail():
            raise RuntimeError("periodic upload failure")
        uploader.submit(fail)
        # Synchronize the test on completion, then exercise the nonblocking trainer poll.
        from concurrent.futures import wait
        wait([uploader._pending], timeout=5)
        try:
            with self.assertRaisesRegex(RuntimeError, "periodic upload failure"):
                uploader.poll()
        finally:
            uploader.close()

    def test_cli_training_overlaps_upload_and_final_flush_keeps_snapshot_step(self):
        engine = _Engine()
        started, progressed = threading.Event(), threading.Event()
        class Session(_Session):
            def step(self):
                if self.engine.global_step == 2:
                    if not started.wait(5):
                        raise AssertionError("upload did not start")
                    progressed.set()
                return super().step()
        session = Session(engine)
        class Coordinator(_Coordinator):
            def publish(self, *args, **kwargs):
                if kwargs["checkpoint_id"] == "step-00000002":
                    started.set()
                    if not progressed.wait(5):
                        raise AssertionError("training blocked on upload")
                return super().publish(*args, **kwargs)
        coordinator = Coordinator(session)
        remote = SimpleNamespace(publisher=object(), drive_manifest={"run_id": "test"},
                                 every_steps=2, rolling_latest_only=True)
        setup = (object(), SimpleNamespace(evaluation_every_steps=0, checkpoint_every_steps=0),
                 engine, session, coordinator)
        output = io.StringIO()
        with patch("trainer.cli.setup", return_value=setup), \
             patch("trainer.cli.configure_remote_publication", return_value=remote), \
             patch("trainer.cli.configure_wandb", return_value=None), \
             patch("trainer.cli.cleanup_remote_publication", return_value=None), \
             patch("trainer.cli.torch.cuda.is_available", return_value=False), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main([*_BASE, "--async-checkpoint-upload"]), 0)
        import json
        events = [json.loads(line)["remote_publication"] for line in output.getvalue().splitlines()
                  if "remote_publication" in line]
        self.assertEqual([event["step"] for event in events], [2, 3])
        self.assertEqual(coordinator.published, ["step-00000002", "step-00000003"])
        self.assertTrue(events[-1]["final"])


# Run the existing best-pointer contracts through the asynchronous trainer path too.
from tests.test_trainer_best_checkpoint import TrainerBestCheckpointTests
from trainer.cli_args import parse_args


class AsyncBestCheckpointTests(TrainerBestCheckpointTests):
    def setUp(self):
        super().setUp()
        parser = patch("trainer.cli.parse_args", side_effect=lambda argv: parse_args([*argv, "--async-checkpoint-upload"]))
        parser.start()
        self.addCleanup(parser.stop)
