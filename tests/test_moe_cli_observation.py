"""Exercise the real process entrypoint, immutable data and joint resume offline."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from trainer.state import load_trainer_state_file
from tests.trainer_fixtures import payload

ROOT = Path(__file__).resolve().parents[1]


class TestMoECLIObservation(unittest.TestCase):
    def test_fresh_process_probe_profile_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            shards = []
            for split, rows in (("train", 4), ("validation", 2)):
                (data / split).mkdir()
                raw = payload([[1+i, 2+i, 3+i, 4+i] for i in range(rows)])
                name = f"{split}/000.bin"
                (data / name).write_bytes(raw)
                shards.append(dict(filename=name, split=split, byte_size=len(raw),
                                   checksum=hashlib.sha256(raw).hexdigest(),
                                   sequence_count=rows, first_block_id=0,
                                   last_block_id=rows//2-1))
            (data / "manifest.json").write_text(json.dumps(dict(
                schema_version=2, sequence_format="context_plus_one", context_length=3,
                stored_tokens_per_sequence=4, sequences_per_block=2, shards=shards)))
            environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                           "OPENBLAS_NUM_THREADS": "2", "WANDB_MODE": "disabled",
                           "HF_HUB_OFFLINE": "1"}

            def run(name, steps, resume=None, extra=()):
                command = [sys.executable, "-m", "MOE_model", "--dataset-dir", str(data),
                           "--checkpoint-dir", str(root/name/"checkpoints"),
                           "--experiment-dir", str(root/name/"artifacts"),
                           "--steps", str(steps), "--model-size", "smoke", "--device", "cpu",
                           "--precision", "fp32", "--microbatch-size", "1",
                           "--load-balancing", "loss_free_sign", "--balancing-step-size", ".001",
                           "--seed", "13", "--probe-sequences", "2", "--validation-blocks", "1",
                           "--checkpoint-at-steps", "0,1,2", "--profile-at-steps", "1", *extra]
                if resume:
                    command += ["--resume", resume]
                result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                        capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-6000:])
                return result

            continuous = run("continuous", 2)
            self.assertEqual(sum("\"validation\"" in line for line in continuous.stdout.splitlines()), 3)
            run("resumed", 1)
            run("resumed", 1, "step-00000001")
            states = [load_trainer_state_file(root/name/"checkpoints/step-00000002/trainer_state.pkl")
                      for name in ("continuous", "resumed")]
            self.assertEqual(states[0]["global_step"], 2)
            self.assertEqual(states[0]["consumed_tokens"], 12)

            def same(first, second):
                if isinstance(first, torch.Tensor):
                    self.assertTrue(torch.equal(first, second))
                elif isinstance(first, dict):
                    self.assertEqual(first.keys(), second.keys())
                    for key in first:
                        same(first[key], second[key])
                elif isinstance(first, (tuple, list)):
                    self.assertEqual(len(first), len(second))
                    for a, b in zip(first, second, strict=True):
                        same(a, b)
                else:
                    self.assertEqual(first, second)
            for key in ("model", "optimizer", "scheduler", "scaler", "consumed_tokens", "torch_rng_state"):
                same(states[0][key], states[1][key])
            artifacts = root/"continuous/artifacts"
            probe = torch.load(artifacts/"probe.pt", weights_only=True)
            self.assertEqual(probe["target_mask"].sum().item(), 6)
            first = torch.load(artifacts/"step-00000000-probe.pt", weights_only=True)
            last = torch.load(artifacts/"step-00000002-probe.pt", weights_only=True)
            self.assertEqual(first["probe_sha256"], last["probe_sha256"])
            self.assertEqual(last["per_token_ce_fp32"].shape, (2, 3))
            self.assertNotIn("lm_logits_fp16", last)
            self.assertEqual(len(last["routing"]), 8)
            trace = next(artifacts.glob("from-*/profile-step-00000001.json")).read_text()
            for label in ("session_step", "forward_ce", "backward", "optimizer", "module/blocks.0.ffn.router"):
                self.assertIn(label, trace)
            identity = json.loads((artifacts/"identity.json").read_text())
            self.assertEqual(identity["model"]["load_balancing"], "loss_free_sign")
            self.assertEqual(len(identity["source_tree_sha256"]), 64)
            events = [json.loads(line) for path in artifacts.glob("from-*/events.jsonl")
                      for line in path.read_text().splitlines()]
            sample = next(event for event in events if event["event"] == "training")
            self.assertTrue(sample["expert_statistics"])
            self.assertTrue(all("gradient_l2_after_clipping" in item
                                for item in sample["expert_statistics"].values()))
            compared = subprocess.run([
                sys.executable, "-m", "trainer.probe_compare",
                str(artifacts/"step-00000000-probe.pt"),
                str(artifacts/"step-00000002-probe.pt"),
                "--checkpoint-before", str(root/"continuous/checkpoints/step-00000000"),
                "--checkpoint-after", str(root/"continuous/checkpoints/step-00000002"),
            ], cwd=ROOT, env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(compared.returncode, 0, compared.stderr)
            comparison = json.loads(compared.stdout)
            self.assertEqual(len(comparison["checkpoint_weight_delta"]["groups"]), 8 * 9)
            self.assertEqual(comparison["ce_delta_after_minus_before"]["count"], 6)
            # Observation and eval must not advance the serialized controller.
            for key, value in states[0]["model"].items():
                if key.endswith("token_exposure"):
                    self.assertEqual(value.sum().item(), 12)

    def test_dataset_verifier_import_does_not_require_torch(self):
        code = '''import sys
class DenyTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('dataset verification imported torch')
sys.meta_path.insert(0, DenyTorch())
import dataset.src.verify
import dataset.incremental_cache
import dataset.incremental_frontier
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
