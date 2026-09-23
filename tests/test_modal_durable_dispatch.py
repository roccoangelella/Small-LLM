"""Exercise real entrypoint control flow with all remote effects replaced."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.test_moe_remote_control import Registry


class DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'modal/moe_production_launch.py'
        spec = importlib.util.spec_from_file_location('durable_entry_test', source)
        cls.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.entry)

    def test_spawn_returns_without_waiting_and_records_handle(self):
        e = self.entry
        cpu = SimpleNamespace(remote=Mock(return_value={'status': 'ready', 'identity': e._production.accepted_identity()}))
        gpu = SimpleNamespace(spawn=Mock(return_value=SimpleNamespace(object_id='fc-test')), remote=Mock())
        registry = Registry()
        with patch.object(e, '_require_source_commit'), patch.object(e, 'prepare_production_cpu', cpu), \
             patch.object(e, 'train_production_h100', gpu), patch.object(e, 'CONTROL', registry), \
             contextlib.redirect_stdout(io.StringIO()):
            e.main(run_id='test-run', dataset_dir='/data/test', steps=100,
                   source_commit='a'*40, resume_source_commit='b'*40,
                   dataset_shard_bucket='owner/data', dataset_shard_run_id='data',
                   checkpoint_bucket='owner/checkpoint', durable=True, segment_id='segment-1')
        cpu.remote.assert_called_once()
        gpu.spawn.assert_called_once()
        gpu.remote.assert_not_called()
        self.assertEqual(registry[('dispatch', 'test-run', 'segment-1')]['call_id'], 'fc-test')

    def test_failed_cpu_gate_never_dispatches_gpu(self):
        e = self.entry
        cpu = SimpleNamespace(remote=Mock(side_effect=RuntimeError('preflight denied')))
        gpu = SimpleNamespace(spawn=Mock(), remote=Mock())
        with patch.object(e, '_require_source_commit'), patch.object(e, 'prepare_production_cpu', cpu), \
             patch.object(e, 'train_production_h100', gpu), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'preflight denied'):
                e.main(run_id='test-run', dataset_dir='/data/test', steps=100,
                       source_commit='a'*40, durable=True, segment_id='segment-1')
        gpu.spawn.assert_not_called()
        gpu.remote.assert_not_called()

    def test_replayed_remote_input_does_not_enter_trainer(self):
        e = self.entry
        control = Mock()
        from moe_remote_control import ClaimBlocked
        control.claim_training.side_effect = ClaimBlocked('attempt already used')
        with patch.object(e, '_runtime_request', return_value=object()), patch.object(e, '_control', return_value=control), \
             patch.object(e._modal, 'current_function_call_id', return_value='fc-test'), \
             patch.object(e._production, 'run_provider_payload') as trainer:
            result = e.train_production_h100.get_raw_f()({}, 'segment-1')
        self.assertEqual(result['status'], 'blocked_retry')
        trainer.assert_not_called()
        control.finish.assert_not_called()

    def test_completed_cpu_preparation_releases_claim(self):
        e = self.entry
        request = SimpleNamespace(run_id='test-run', resume='latest', checkpoint_bucket='owner/bucket', streaming=True)
        control = Mock()
        control.checkpoints.read_json.return_value = {'checkpoint_id': 'step-00000100'}
        volume = Mock()
        with patch.object(e, '_runtime_request', return_value=request), patch.object(e, '_control', return_value=control), \
             patch.object(e, 'RUN_VOLUME', volume), patch.object(e, '_dataset_volume', return_value=volume), \
             patch.object(e._production, 'prepare_dataset', return_value={'status': 'ready', 'training_complete': True}):
            result = e.prepare_production_cpu.get_raw_f()({}, 'segment-1')
        self.assertTrue(result['training_complete'])
        control.finish.assert_called_once_with('segment-1', {'status': 'complete', 'training_complete': True})

    def test_registry_outage_is_not_reported_as_replay_success(self):
        e = self.entry
        control = Mock()
        control.claim_training.side_effect = OSError('registry unavailable')
        with patch.object(e, '_runtime_request', return_value=object()), patch.object(e, '_control', return_value=control), \
             patch.object(e._modal, 'current_function_call_id', return_value='fc-test'), \
             patch.object(e._production, 'run_provider_payload') as trainer:
            with self.assertRaisesRegex(OSError, 'registry unavailable'):
                e.train_production_h100.get_raw_f()({}, 'segment-1')
        trainer.assert_not_called()

    def test_auto_bucket_uses_existing_resolver(self):
        e = self.entry
        request = SimpleNamespace(checkpoint_bucket='auto', dataset_shard_bucket='owner/data')
        with patch.dict('os.environ', {'SMALL_LLM_HF_CHECKPOINT_BUCKET_ID': 'owner/actual'}), \
             patch('moe_remote_control.RemoteControl'), \
             patch('dataset.src.hf_bucket_checkpoint.HuggingFaceBucketCheckpointStore') as checkpoint_store, \
             patch('dataset.src.hf_bucket_shards.HuggingFaceBucketShardStore'):
            e._control(request)
        self.assertEqual(checkpoint_store.call_args.args[0], 'owner/actual')


if __name__ == '__main__':
    unittest.main()
