"""Fast, provider-free checks for the bounded 100M/2B pilot protocol."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))
import moe_pilot as pilot  # noqa: E402


def _request(root: Path, *, arm: str = "M1", provider: str = "modal", steps: int = 100) -> object:
    return pilot.PilotRequest(
        provider=provider,
        arm=arm,
        run_id="pilot-test-001",
        dataset_dir=root / "dataset",
        run_root=root / "runs",
        steps=steps,
        precision="bf16",
        microbatch_size=16,
        source_commit="a" * 40,
        gamma=0.001 if arm == "M1" else None,
    )


def _plan() -> object:
    return pilot.PilotPlan(
        manifest_path=Path("/data/manifest.json"),
        manifest_sha256="b" * 64,
        trainer={
            "steps": 1000,
            "full_block_target_tokens": 131072,
            "schedule": "wsd",
            "warmup_tokens": 655360,
            "stable_tokens": 9830400,
            "decay_tokens": 2621440,
            "minimum_lr_ratio": 0.1,
            "validation_blocks": 16,
        },
        steps=1000,
        sequences_per_block=64,
    )


class _TrainerState:
    def __init__(self, step: int, tokens: int) -> None:
        self.step, self.tokens = step, tokens

    def state_dict(self):
        return {"global_step": self.step, "consumed_tokens": self.tokens}


class PilotProtocolTests(unittest.TestCase):
    def test_command_selects_dense_or_moe_module_and_freezes_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, module, expected in (
                ("D", "trainer", ()),
                ("M0", "MOE_model", ("none", "0")),
                ("M1", "MOE_model", ("loss_free_sign", "0.001")),
            ):
                request = _request(root, arm=arm)
                spec = pilot.PilotSpec(request, _plan(), root / "ns", root / "ckpt", root / "exp",
                                       (0, 25, 100), (50,))
                command = pilot.build_pilot_command(spec, remaining_steps=75, resume="step-00000025")
                self.assertEqual(command[2], module)
                self.assertEqual(command[command.index("--steps") + 1], "75")
                self.assertEqual(command[command.index("--resume") + 1], "step-00000025")
                if expected:
                    self.assertEqual(command[command.index("--load-balancing") + 1], expected[0])
                    self.assertEqual(command[command.index("--balancing-step-size") + 1], expected[1])
                else:
                    self.assertNotIn("--load-balancing", command)
                self.assertEqual(command[command.index("--gdn-chunk-size") + 1], "32")

    def test_short_pilot_keeps_real_full_manifest_schedule(self):
        from types import SimpleNamespace
        from tests.test_dataset_qualification import _manifest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dataset"
            data.mkdir()
            (data / "manifest.json").write_text(json.dumps(_manifest("modal-2b-b64", 15267)))
            # Data verifier has its own tests; exercise the real schedule derivation here.
            with patch.object(pilot, "verify", return_value=SimpleNamespace(passed=True, complete=True)):
                spec = pilot.prepare_pilot(_request(root, steps=100))
            self.assertEqual(spec.plan.steps, 15267)
            self.assertEqual(spec.plan.trainer["warmup_tokens"], 764 * 131072)
            self.assertEqual(sum(spec.plan.trainer[key] for key in
                                 ("warmup_tokens", "stable_tokens", "decay_tokens")), 15267 * 131072)
            self.assertEqual(spec.checkpoint_steps, (0, 25, 50, 100))

    def test_m1_requires_explicit_gamma_and_caps_are_provider_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "explicit gamma"):
                pilot.validate_request(pilot.PilotRequest(
                    provider="modal", arm="M1", run_id="safe", dataset_dir=root,
                    run_root=root, steps=1, precision="fp16", microbatch_size=1,
                    source_commit="a" * 40,
                ))
            with self.assertRaisesRegex(ValueError, "exceeds 300"):
                pilot.validate_request(_request(root, steps=301))
            with self.assertRaisesRegex(ValueError, "exceeds 100"):
                pilot.validate_request(_request(root, arm="D", steps=101))

    def test_payload_round_trip_does_not_import_trainer_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            payload = pilot.request_payload(request)
            restored = pilot.request_from_payload(payload, provider="modal", run_root=root / "runs")
            self.assertEqual(restored, request)

    def test_default_profiles_are_filtered_to_the_cap(self) -> None:
        self.assertEqual(pilot._selected(25, None, True), ())
        self.assertEqual(pilot._selected(50, None, True), (50,))
        self.assertEqual(pilot._selected(25, None, False), (0, 25))

    def test_frozen_namespace_rejects_changed_arm_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            with patch.object(pilot, "_plan_from_manifest", return_value=_plan()):
                first = pilot.prepare_pilot(request)
                changed = pilot.PilotRequest(
                    provider=request.provider, arm=request.arm, run_id=request.run_id,
                    dataset_dir=request.dataset_dir, run_root=request.run_root, steps=request.steps,
                    precision=request.precision, microbatch_size=request.microbatch_size,
                    source_commit=request.source_commit, gamma=0.002,
                )
                self.assertEqual(first.namespace, root / "runs" / "moe-pilots" / "modal" / "pilot-test-001" / "M1")
                with self.assertRaisesRegex(RuntimeError, "different frozen"):
                    pilot.prepare_pilot(changed)

    def test_latest_checkpoint_uses_checkpoint_state_token_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from dataset.src.joint_checkpoint import CheckpointCoordinator

            coordinator = CheckpointCoordinator(root, configuration_hash="c", source_hash="s", schema_hash="h")
            state = _TrainerState(3, 777)
            checkpoint = coordinator.save(
                checkpoint_id="step-00000003", trainer=state,
                pipeline_state={"last_consumed_block_id": 2, "gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            latest = pilot.find_latest_complete_checkpoint(root)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["path"], checkpoint)
            self.assertEqual(latest["consumed_tokens"], 777)

    def test_child_checkpoint_event_is_verified_before_callback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root, steps=2)
            with patch.object(pilot, "_plan_from_manifest", return_value=_plan()):
                spec = pilot.prepare_pilot(request)
                from dataset.src.joint_checkpoint import CheckpointCoordinator

                coordinator = CheckpointCoordinator(spec.checkpoint_dir, configuration_hash="c", source_hash="s", schema_hash="h")
                coordinator.save(
                    checkpoint_id="step-00000001", trainer=_TrainerState(1, 123),
                    pipeline_state={"last_consumed_block_id": 0, "gradient_accumulation_position": 0},
                    optimizer_step_complete=True,
                )
                seen: list[str] = []

                class Process:
                    stdout = io.StringIO(json.dumps({"local_checkpoint": {"checkpoint_id": "step-00000001"}}) + "\n")
                    def wait(self): return 0
                    def kill(self): raise AssertionError("child must not be killed")

                with self.assertRaisesRegex(RuntimeError, "stopped before"):
                    pilot.run_pilot(
                        request,
                        checkpoint_callback=lambda checkpoint_id, _path: seen.append(checkpoint_id),
                        popen_factory=lambda *args, **kwargs: Process(),
                    )
                self.assertEqual(seen, ["step-00000001"])

    def test_announced_checkpoint_requires_consistent_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(pilot, "_plan_from_manifest", return_value=_plan()):
                spec = pilot.prepare_pilot(_request(root, steps=2))
            from dataset.src.joint_checkpoint import CheckpointCoordinator
            coordinator = CheckpointCoordinator(spec.checkpoint_dir, configuration_hash="c", source_hash="s", schema_hash="h")
            coordinator.save(
                checkpoint_id="step-00000001", trainer=_TrainerState(2, 123),
                pipeline_state={"last_consumed_block_id": 0, "gradient_accumulation_position": 0},
                optimizer_step_complete=True,
            )
            with self.assertRaisesRegex(RuntimeError, "invalid checkpoint"):
                pilot._event_checkpoint(spec, {"checkpoint_id": "step-00000001"})

    def test_failed_child_persists_attempt_runtime_and_final_callback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root, steps=1)
            with patch.object(pilot, "_plan_from_manifest", return_value=_plan()):
                final_calls: list[bool] = []

                class Process:
                    stdout = io.StringIO("child failed\n")
                    def wait(self): return 9
                    def kill(self): return None

                with self.assertRaisesRegex(RuntimeError, "status 9"):
                    pilot.run_pilot(
                        request,
                        final_callback=lambda: final_calls.append(True),
                        popen_factory=lambda *args, **kwargs: Process(),
                    )
                runtime_path = root / "runs" / "moe-pilots" / "modal" / "pilot-test-001" / "M1" / "runtime.json"
                self.assertGreater(json.loads(runtime_path.read_text())["seconds"], 0)
                self.assertEqual(final_calls, [True])


class ProviderSourceTests(unittest.TestCase):
    def test_provider_files_are_sdk_wrappers_without_trainer_monkeypatch(self) -> None:
        for relative in ("modal/moe_launch.py", "beam/moe_launch.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
            self.assertNotIn("trainer_cli.setup", source)
            self.assertIn("_local_source_commit", source)
            self.assertIn("request_payload", source)
            self.assertIn("prepare_provider_payload", source)
            self.assertIn("run_provider_payload", source)
            self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "prepare_pilot_cpu" for node in tree.body))
        modal = (ROOT / "modal/moe_launch.py").read_text(encoding="utf-8")
        beam = (ROOT / "beam/moe_launch.py").read_text(encoding="utf-8")
        self.assertIn('gpu="H100"', modal)
        self.assertIn('gpu="RTX4090"', beam)
        self.assertIn("RUN_VOLUME.commit", modal)
        self.assertIn("NOOP_VOLUME", beam)
        self.assertLess(modal.index("prepare_pilot_cpu.remote"), modal.index("train_pilot_h100.remote"))
        self.assertLess(beam.index("prepare_pilot_cpu.remote"), beam.index("train_pilot_rtx4090.remote"))



class ProviderExecutableTests(unittest.TestCase):
    def test_import_and_dispatch_real_wrappers_with_sdk_boundary_stubbed(self):
        import subprocess
        script = r'''
import runpy, sys, types
from unittest.mock import MagicMock
provider = sys.argv[1]
def decorate(**options):
    return lambda function: function
class App:
    def __init__(self, *args, **kwargs): pass
    function = staticmethod(decorate)
    local_entrypoint = staticmethod(decorate)
sdk = types.ModuleType(provider)
sdk.Image = MagicMock()
sdk.Volume = MagicMock()
sdk.Secret = MagicMock()
sdk.Retries = MagicMock()
sdk.App = App
sdk.function = decorate
sys.modules[provider] = sdk
module = runpy.run_path(provider + '/moe_launch.py')
module['_base']._local_source_commit = lambda: 'a' * 40
order = []
def prepared(payload):
    order.append(('prepare', payload))
    return {'status': 'ready'}
def trained(payload):
    order.append(('train', payload))
    return {'status': 'complete'}
module['prepare_pilot_cpu'].remote = prepared
module['train_pilot_h100' if provider == 'modal' else 'train_pilot_rtx4090'].remote = trained
if provider == 'modal':
    module['main'](arm='M1', run_id='wrapper-test', dataset_dir='/data/ready',
                   precision='fp16', microbatch_size=4, source_commit='a'*40,
                   steps=100, gamma=.001)
else:
    module['main'](['--arm', 'M1', '--run-id', 'wrapper-test', '--dataset-dir', '/data/ready',
                    '--precision', 'fp16', '--microbatch-size', '4', '--source-commit', 'a'*40,
                    '--steps', '100', '--gamma', '.001'])
assert [row[0] for row in order] == ['prepare', 'train'], order
assert order[0][1] == order[1][1]
assert 'torch' not in sys.modules, 'CPU launcher imported torch'
'''
        for provider in ("modal", "beam"):
            with self.subTest(provider=provider):
                result = subprocess.run([sys.executable, "-c", script, provider], cwd=ROOT,
                                        text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
