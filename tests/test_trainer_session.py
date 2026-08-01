import unittest
import torch
from trainer import LiveBlockConsumer, TrainerConfig, TrainerEngine, TrainingSession, generate_token_ids
from tests.trainer_fixtures import Coordinator, PreparedBlock, TinyLM, batch, payload

class TrainerSessionTests(unittest.TestCase):
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
