import unittest
from unittest.mock import Mock, patch
import torch
from trainer import LiveBlockConsumer, TrainerConfig, TrainerEngine, TrainingSession, generate_token_ids
from tests.trainer_fixtures import Coordinator, PreparedBlock, TinyLM, batch, payload

class TrainerSessionTests(unittest.TestCase):
    def test_late_optimizer_failure_is_not_retried_or_acknowledged(self):
        from MOE_model.engine import MoETrainerEngine
        from MOE_model.model import MoESmallLLM
        from tests.test_moe_model import _tiny_moe_config

        for kind in ("dense", "moe"):
            with self.subTest(kind=kind):
                model = TinyLM() if kind == "dense" else MoESmallLLM(_tiny_moe_config())
                engine_type = TrainerEngine if kind == "dense" else MoETrainerEngine
                engine = engine_type(model, TrainerConfig(precision="fp32",
                    optimizer="adamw" if kind == "dense" else "hybrid_muon_adamw",
                    microbatch_size=1, weight_decay=0.0), device="cpu")
                source = Mock()
                source.next_batch.return_value = batch(0)
                session = TrainingSession(engine, source)
                parameter = next(model.parameters())
                before = parameter.detach().clone()

                def fail_after_mutation(*args, **kwargs):
                    with torch.no_grad():
                        parameter.add_(1)
                    raise FloatingPointError("late optimizer failure")

                with patch.object(engine.optimizer, "step", side_effect=fail_after_mutation) as step:
                    with self.assertRaisesRegex(FloatingPointError, "late optimizer failure"):
                        session.step()
                step.assert_called_once()
                source.acknowledge.assert_not_called()
                self.assertFalse(torch.equal(parameter, before))
                self.assertEqual(engine.global_step, 0)
                self.assertEqual(engine.consumed_tokens, 0)
                self.assertEqual(engine.scheduler.committed_tokens, 0)
                self.assertEqual(engine.overflow_events, 0)
                if kind == "moe":
                    for block in model.blocks:
                        self.assertEqual(int(block.ffn.router.token_exposure.sum()), 0)
                        self.assertEqual(int(block.ffn.router.expert_updates.sum()), 0)

    def test_joint_checkpoint_cursor_and_generation(self):
        blocks = [PreparedBlock(0,"train",2,8,payload([[1,2,3,4],[2,3,4,5]])),
                  PreparedBlock(1,"train",2,8,payload([[2,3,4,5],[3,4,5,6]]))]
        consumer = LiveBlockConsumer(2, context_length=3, semantic_vocab_size=16)
        for block in blocks: consumer.submit(block)
        engine = TrainerEngine(TinyLM(), TrainerConfig(precision="fp32",
            microbatch_size=1, weight_decay=0.0), device="cpu")
        session = TrainingSession(engine, consumer)
        session.step(); self.assertEqual(consumer.last_acknowledged_block_id, 0)
        with self.assertRaises(RuntimeError):
            session.save_checkpoint(Coordinator(), "step-1")
        session.step(); coordinator = Coordinator()
        session.save_checkpoint(coordinator, "step-2")
        restored_consumer = LiveBlockConsumer(2, context_length=3,
            semantic_vocab_size=16, last_consumed_block_id=-1)
        restored = TrainingSession(TrainerEngine(TinyLM(), engine.config, device="cpu"),
                                   restored_consumer)
        restored.load_checkpoint(coordinator, "step-2")
        self.assertEqual(restored_consumer.last_acknowledged_block_id, 1)
        self.assertEqual(restored.engine.global_step, 2)
        metrics = engine.evaluate([batch(0, split="validation")])
        self.assertGreater(metrics["loss"], 0)
        generated = generate_token_ids(engine.model, torch.tensor([[1,2]]),
                                       max_new_tokens=3, max_seq_len=4)
        self.assertEqual(generated.shape, (1,5))
