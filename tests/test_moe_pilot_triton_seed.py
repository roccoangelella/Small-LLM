"""Pilot runner × Triton seed (ADR 0169): env for the child, status file, harvest and cache commit. No GPU."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import moe_pilot as pilot  # noqa: E402
from tests.test_moe_pilot_protocol import _plan, _request  # noqa: E402


def _contract() -> dict[str, object]:
    return {
        "gpu_name": "NVIDIA H100 80GB HBM3", "compute_capability": [9, 0], "python": "3.13", "torch": "2.10.0",
        "cuda": "12.8", "triton": "3.6.0", "fla_core": "0.5.2", "model": "100M", "architecture": "gdn2_hybrid",
        "precision": "bf16", "microbatch_size": 16, "context_length": 2048, "gdn_chunk_size": 32,
        "kernel_contract_sha256": "d" * 64,
    }


class _Process:
    """Fake child: 'compiles' a kernel into the cache dir it was given, announces no checkpoint, exits 0."""

    def __init__(self, env: dict[str, str], step: int) -> None:
        self.env, self.step = env, step
        self.stdout = io.StringIO("child ok\n")

    def wait(self) -> int:
        if "TRITON_CACHE_DIR" in self.env:  # a real child would compile into the seeded/JIT cache dir
            cache_dir = Path(self.env["TRITON_CACHE_DIR"])
            (cache_dir / "k1").mkdir(parents=True, exist_ok=True)
            (cache_dir / "k1" / "kernel.cubin").write_bytes(b"sass")
        return 0

    def kill(self) -> None:
        return None


class PilotTritonSeedTests(unittest.TestCase):
    def _run(self, root: Path, *, seed_root: Path | None, contract, local_root: Path, commit_error=False):
        request = _request(root, steps=1)
        seen: dict[str, object] = {}
        commits: list[bool] = []

        def popen(command, **kwargs):
            seen["env"] = dict(kwargs["env"])
            return _Process(kwargs["env"], 1)

        def commit():
            if commit_error:
                raise OSError("cache volume unavailable")
            commits.append(True)

        seed_module = pilot._triton_seed_module()
        with (
            patch.object(pilot, "_plan_from_manifest", return_value=_plan()),
            patch.object(pilot, "_result", return_value={"completed_steps": 1, "checkpoint_id": "step-00000001"}),
            patch.object(pilot, "_triton_seed_module", return_value=seed_module),
            patch.object(seed_module, "live_contract", return_value=contract),
            patch.object(seed_module, "LOCAL_ROOT", local_root),
        ):
            # prepare()/harvest() default local_root is read at call time from the module attribute.
            original_prepare, original_harvest = seed_module.prepare, seed_module.harvest
            seed_module.prepare = lambda **kw: original_prepare(local_root=local_root, **kw)
            seed_module.harvest = lambda **kw: original_harvest(local_root=local_root, **kw)
            pilot.run_pilot(request, popen_factory=popen, triton_seed_root=seed_root,
                            cache_commit=commit)
        namespace = root / "runs" / "moe-pilots" / "modal" / "pilot-test-001" / "M1"
        status = json.loads((namespace / "experiment" / "triton_seed.json").read_text())
        return seen["env"], status, commits

    def test_first_run_falls_back_to_jit_then_publishes_the_seed_and_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_root, local_root = root / "cache" / "triton", root / "tmp-triton"
            env, status, commits = self._run(root, seed_root=seed_root, contract=_contract(), local_root=local_root)
            self.assertEqual(status["before"]["status"], "jit_fallback")
            self.assertEqual(env["TRITON_CACHE_DIR"], status["before"]["cache_dir"])
            self.assertEqual(env["TRITON_CACHE_AUTOTUNING"], "1")
            self.assertEqual(env["FLA_CACHE_RESULTS"], "1")
            self.assertEqual(status["after"]["status"], "published")
            self.assertEqual(len(list(Path(status["after"]["seed_dir"]).glob("triton-cache-*.tar"))), 1)
            self.assertEqual(commits, [True])
            # Second cold container (local cache wiped): the seed is found, nothing is re-published, no commit.
            import shutil
            shutil.rmtree(local_root)
            env2, status2, commits2 = self._run(root, seed_root=seed_root, contract=_contract(), local_root=local_root)
            self.assertEqual(status2["before"]["status"], "seeded")
            self.assertNotIn("after", status2)
            self.assertEqual(commits2, [])
            self.assertTrue((Path(env2["TRITON_CACHE_DIR"]) / "k1" / "kernel.cubin").is_file())

    def test_cache_commit_failure_does_not_fail_successful_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, status, commits = self._run(root, seed_root=root / "cache", contract=_contract(),
                                           local_root=root / "local", commit_error=True)
            self.assertEqual(status["after"]["status"], "error")
            self.assertIn("cache volume unavailable", status["after"]["reason"])
            self.assertGreaterEqual(status["after"]["harvest_and_commit_seconds"], 0)
            self.assertEqual(commits, [])

    def test_dense_and_moe_do_not_claim_the_same_seed_contract(self):
        from unittest.mock import Mock
        module = Mock()
        root = Path("/tmp/unused")
        pilot._triton_seed_contract(_request(root, arm="D"), module)
        dense = module.live_contract.call_args.kwargs
        pilot._triton_seed_contract(_request(root, arm="M0"), module)
        moe = module.live_contract.call_args.kwargs
        self.assertNotEqual(dense["model"], moe["model"])

    def test_without_a_gpu_query_the_seed_is_disabled_and_the_env_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, status, commits = self._run(root, seed_root=root / "cache" / "triton", contract=None,
                                             local_root=root / "tmp-triton")
            self.assertEqual(status["before"]["status"], "disabled")
            self.assertNotIn("TRITON_CACHE_DIR", env)
            self.assertEqual(commits, [])
            self.assertFalse((root / "cache").exists())

    def test_pilot_identity_does_not_mention_the_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root, steps=1)
            with patch.object(pilot, "_plan_from_manifest", return_value=_plan()):
                spec = pilot.prepare_pilot(request)
            contract = json.loads((spec.namespace / "pilot_contract.json").read_text())
            self.assertNotIn("triton", json.dumps(contract).lower())


if __name__ == "__main__":
    unittest.main()
