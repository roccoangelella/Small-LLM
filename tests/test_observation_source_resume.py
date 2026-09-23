"""Existing observation roots survive an explicit production executor migration."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from trainer.observation import RunObservation

class ObservationSourceResume(unittest.TestCase):
    def test_explicit_origin_preserves_root_and_records_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{}')
            args = SimpleNamespace(experiment_dir=root/'observations', dataset_manifest=root/'manifest.json',
                dataset_dir=root, initialization='normal', sequences_per_block=None, source_commit='origin',
                resume=None, run_source_commit=None, profile_at_steps=[], checkpoint_at_steps=[],
                probe_sequences=0, probe_lm_logits=False)
            model = torch.nn.Linear(2, 2)
            engine = SimpleNamespace(model=model, optimizer=torch.optim.AdamW(model.parameters()),
                device=torch.device('cpu'), global_step=50000, consumed_tokens=100)
            config = SimpleNamespace(version=3, as_dict=lambda: {'version': 3, 'width': 2})
            recipe = {'learning_rate': .001, 'microbatch_size': 32}
            trainer = SimpleNamespace(as_dict=lambda: dict(recipe))
            def create():
                source = {'source_commit': args.source_commit, 'source_tree_sha256': args.source_commit,
                          'source_tree_dirty': False}
                with patch('trainer.observation._source_identity', return_value=source), contextlib.redirect_stdout(io.StringIO()):
                    return RunObservation(args, engine, config, trainer, None)
            create()
            identity = root/'observations/identity.json'
            before = identity.read_bytes()
            args.source_commit = 'executor'
            with self.assertRaises(ValueError): create()
            args.resume = 'step-00050000'
            with self.assertRaises(ValueError): create()
            args.run_source_commit = 'wrong'
            with self.assertRaisesRegex(ValueError, 'origin'): create()
            args.run_source_commit = 'origin'
            observation = create()
            self.assertEqual(identity.read_bytes(), before)
            manifest = json.loads((observation.segment/'manifest.json').read_text())
            self.assertEqual(manifest['source_commit'], 'executor')
            self.assertEqual(manifest['run_source_commit'], 'origin')
            recipe['learning_rate'] = .002
            with self.assertRaises(ValueError): create()
            recipe['learning_rate'] = .001
            (root/'manifest.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): create()
            self.assertEqual(identity.read_bytes(), before)
