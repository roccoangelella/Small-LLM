import unittest
import torch
from trainer import LiveBlockConsumer, PreparedBlockDecoder
from tests.trainer_fixtures import PreparedBlock, payload

class LiveConsumerTests(unittest.TestCase):
    def test_decode_and_acknowledge_only_after_step_boundary(self):
        block = PreparedBlock(0, "train", 2, 8,
            payload([[1, 2, 3, 4], [5, 6, 7, 8]]))
        batch = PreparedBlockDecoder(context_length=3, semantic_vocab_size=16).decode(block)
        self.assertTrue(torch.equal(batch.input_ids, torch.tensor([[1, 2, 3], [5, 6, 7]])))
        self.assertTrue(torch.equal(batch.labels, torch.tensor([[2, 3, 4], [6, 7, 8]])))
        consumer = LiveBlockConsumer(2, context_length=3, semantic_vocab_size=16)
        consumer.submit(block)
        received = consumer.next_batch()
        with self.assertRaises(RuntimeError):
            consumer.pipeline_state()
        consumer.acknowledge(received.block_id)
        self.assertEqual(consumer.pipeline_state()["last_consumed_block_id"], 0)
