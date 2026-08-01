import pickle, unittest
import torch
from trainer import TrainerConfig, TrainerEngine
from tests.trainer_fixtures import TinyLM, batch

class TrainerResumeTests(unittest.TestCase):
    def test_state_round_trip_matches_uninterrupted_training(self):
        torch.manual_seed(7)
        initial = TinyLM().state_dict()
        config = TrainerConfig(precision="fp32", microbatch_size=1,
                               learning_rate=1e-2, weight_decay=0.0)
        baseline_model = TinyLM(); baseline_model.load_state_dict(initial)
        baseline = TrainerEngine(baseline_model, config, device="cpu")
        baseline.train_batch(batch(0)); baseline.train_batch(batch(1, 1))
        interrupted_model = TinyLM(); interrupted_model.load_state_dict(initial)
        interrupted = TrainerEngine(interrupted_model, config, device="cpu")
        interrupted.train_batch(batch(0))
        state = pickle.loads(pickle.dumps(interrupted.state_dict()))
        resumed = TrainerEngine(TinyLM(), config, device="cpu")
        resumed.load_state_dict(state); resumed.train_batch(batch(1, 1))
        self.assertEqual((resumed.global_step, resumed.consumed_tokens), (2, 12))
        for expected, actual in zip(baseline.model.parameters(), resumed.model.parameters(), strict=True):
            self.assertTrue(torch.equal(expected, actual))
