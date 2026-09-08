"""Provider-agnostic Triton seed: package, verify, extract, fallback, harvest (no GPU needed)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trainer import triton_seed


def _contract(**overrides: object) -> dict[str, object]:
    base = {
        "gpu_name": "NVIDIA H100 80GB HBM3", "compute_capability": [9, 0], "python": "3.13",
        "torch": "2.10.0", "cuda": "12.8", "triton": "3.6.0", "fla_core": "0.5.2", "model": "100M",
        "architecture": "gdn2_hybrid", "precision": "fp16", "microbatch_size": 8, "context_length": 2048,
        "gdn_chunk_size": 32, "kernel_contract_sha256": "c" * 64,
    }
    base.update(overrides)
    return base


def _populate(cache_dir: Path) -> None:
    (cache_dir / "abc123").mkdir(parents=True)
    (cache_dir / "abc123" / "kernel.cubin").write_bytes(b"\x00cubin" * 100)
    (cache_dir / "abc123" / "kernel.json").write_text('{"name": "kernel"}', encoding="utf-8")
    (cache_dir / "abc123" / "tmp.pid_1_x").write_bytes(b"inflight")  # must be excluded
    (cache_dir / "lock").write_text("", encoding="utf-8")            # must be excluded


class TritonSeedTests(unittest.TestCase):
    def test_cache_id_is_deterministic_and_geometry_specific(self) -> None:
        a, b = triton_seed.cache_id(_contract()), triton_seed.cache_id(_contract(microbatch_size=16))
        self.assertTrue(a.startswith("sm90-py3.13-torch2.10.0-cu12.8-triton3.6.0-fla0.5.2-100M-gdn2_hybrid-fp16-mb8-"))
        self.assertNotEqual(a, b)
        self.assertEqual(a, triton_seed.cache_id(_contract()))

    def test_seed_built_at_another_canonical_path_is_rejected(self) -> None:
        # Triton metadata stores absolute paths (ADR 0102): a seed is only valid at the path it was built at.
        with tempfile.TemporaryDirectory() as temporary:
            root, contract = Path(temporary), _contract()
            _populate(triton_seed.local_cache_dir(contract, root / "a"))
            triton_seed.harvest(contract=contract, seed_root=root / "volume", local_root=root / "a")
            status = triton_seed.prepare(contract=contract, seed_root=root / "volume", local_root=root / "b")
            self.assertEqual(status["status"], "jit_fallback")
            self.assertIn("canonical path", status["rejection"])

    def test_round_trip_seed_restores_bytes_in_fresh_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, contract = Path(temporary), _contract()
            local_root, seed_root = root / "local", root / "volume" / "triton"
            cache_dir = triton_seed.local_cache_dir(contract, local_root)
            _populate(cache_dir)
            published = triton_seed.harvest(contract=contract, seed_root=seed_root, local_root=local_root)
            self.assertEqual(published["status"], "published")
            self.assertEqual(published["files"], 2)  # cubin + json, scratch excluded
            seed_dir = Path(published["seed_dir"])
            self.assertTrue((seed_dir / triton_seed.MANIFEST_NAME).is_file())
            self.assertEqual(len(list(seed_dir.glob("triton-cache-*.tar"))), 1)
            # A fresh container: same canonical local path, nothing compiled yet, the seed is on the volume.
            import shutil
            shutil.rmtree(cache_dir)
            fresh_local = local_root
            status = triton_seed.prepare(contract=contract, seed_root=seed_root, local_root=fresh_local)
            self.assertEqual(status["status"], "seeded")
            extracted = Path(status["cache_dir"])
            self.assertEqual((extracted / "abc123" / "kernel.cubin").read_bytes(), b"\x00cubin" * 100)
            self.assertFalse((extracted / "abc123" / "tmp.pid_1_x").exists())
            self.assertEqual(status["env"]["TRITON_CACHE_DIR"], str(extracted))
            self.assertEqual(status["env"]["TRITON_CACHE_AUTOTUNING"], "1")
            # Second prepare in the same container reuses the validated local seed.
            again = triton_seed.prepare(contract=contract, seed_root=seed_root, local_root=fresh_local)
            self.assertEqual(again["status"], "local_seed")
            # Harvest does not overwrite an existing seed.
            self.assertEqual(triton_seed.harvest(contract=contract, seed_root=seed_root, local_root=fresh_local)["status"], "exists")

    def test_incompatible_seed_falls_back_to_jit_and_strict_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_root, seed_root = root / "local", root / "volume" / "triton"
            built = _contract()
            _populate(triton_seed.local_cache_dir(built, local_root))
            triton_seed.harvest(contract=built, seed_root=seed_root, local_root=local_root)
            # Same cache id on disk but a tampered manifest contract (e.g. a kernel-source change with the same id).
            seed_dir = seed_root / triton_seed.cache_id(built)
            manifest = json.loads((seed_dir / triton_seed.MANIFEST_NAME).read_text())
            manifest["contract"]["fla_core"] = "0.5.3"
            (seed_dir / triton_seed.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
            import shutil
            shutil.rmtree(triton_seed.local_cache_dir(built, local_root))
            status = triton_seed.prepare(contract=built, seed_root=seed_root, local_root=local_root)
            self.assertEqual(status["status"], "jit_fallback")
            self.assertIn("contract", status["rejection"])
            self.assertTrue(Path(status["cache_dir"]).is_dir())  # JIT target exists for a later harvest
            shutil.rmtree(triton_seed.local_cache_dir(built, local_root))
            with self.assertRaises(triton_seed.TritonSeedError):
                triton_seed.prepare(contract=built, seed_root=seed_root, local_root=local_root, strict=True)

    def test_corrupt_archive_is_rejected_and_nothing_is_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, contract = Path(temporary), _contract()
            local_root, seed_root = root / "local", root / "volume" / "triton"
            _populate(triton_seed.local_cache_dir(contract, local_root))
            seed_dir = Path(triton_seed.harvest(contract=contract, seed_root=seed_root, local_root=local_root)["seed_dir"])
            with next(seed_dir.glob("triton-cache-*.tar")).open("ab") as handle:
                handle.write(b"garbage")
            import shutil
            shutil.rmtree(triton_seed.local_cache_dir(contract, local_root))
            status = triton_seed.prepare(contract=contract, seed_root=seed_root, local_root=local_root)
            self.assertEqual(status["status"], "jit_fallback")
            self.assertIn("size mismatch", status["rejection"])
            self.assertFalse((Path(status["cache_dir"]) / "abc123").exists())

    def test_no_gpu_or_disable_env_means_disabled_and_no_env(self) -> None:
        self.assertEqual(triton_seed.prepare(contract=None, seed_root=None)["status"], "disabled")
        with patch.dict(os.environ, {triton_seed.DISABLE_ENV: "1"}):
            status = triton_seed.prepare(contract=_contract(), seed_root=None)
        self.assertEqual(status["status"], "disabled")
        self.assertEqual(status["env"], {})

    def test_kernel_contract_hash_tracks_kernel_facing_sources_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            for relative in triton_seed.KERNEL_CONTRACT_FILES:
                (repo / relative).parent.mkdir(parents=True, exist_ok=True)
                (repo / relative).write_text("x = 1\n", encoding="utf-8")
            first = triton_seed.kernel_contract_sha256(repo)
            (repo / "kaggle").mkdir()
            (repo / "kaggle" / "wrapper.py").write_text("y = 2\n", encoding="utf-8")
            self.assertEqual(first, triton_seed.kernel_contract_sha256(repo))
            (repo / "model" / "gdn2_fla.py").write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(first, triton_seed.kernel_contract_sha256(repo))

    def test_strict_requires_a_device_and_cannot_be_disabled(self):
        with self.assertRaises(triton_seed.TritonSeedError):
            triton_seed.prepare(contract=None, seed_root=None, strict=True)
        with patch.dict(os.environ, {triton_seed.DISABLE_ENV: "1"}):
            with self.assertRaises(triton_seed.TritonSeedError):
                triton_seed.prepare(contract=_contract(), seed_root=None, strict=True)

    def test_malformed_seed_falls_back_and_is_repaired_after_jit(self):
        with tempfile.TemporaryDirectory() as temporary:
            import shutil
            root, contract = Path(temporary), _contract()
            local, volume = root / "local", root / "volume"
            cache = triton_seed.local_cache_dir(contract, local)
            _populate(cache)
            seed = Path(triton_seed.harvest(contract=contract, seed_root=volume, local_root=local)["seed_dir"])
            manifest_path = seed / triton_seed.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["archive"]["size"] = "invalid"
            manifest_path.write_text(json.dumps(manifest))
            shutil.rmtree(cache)
            status = triton_seed.prepare(contract=contract, seed_root=volume, local_root=local)
            self.assertEqual(status["status"], "jit_fallback")
            _populate(cache)
            self.assertEqual(triton_seed.harvest(contract=contract, seed_root=volume, local_root=local)["status"], "published")
            shutil.rmtree(cache)
            self.assertEqual(triton_seed.prepare(contract=contract, seed_root=volume, local_root=local, strict=True)["status"], "seeded")

    def test_deleted_local_kernel_is_restored_from_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, contract = Path(temporary), _contract()
            local, volume = root / "local", root / "volume"
            cache = triton_seed.local_cache_dir(contract, local)
            _populate(cache)
            triton_seed.harvest(contract=contract, seed_root=volume, local_root=local)
            (cache / "abc123" / "kernel.cubin").unlink()
            self.assertEqual(triton_seed.prepare(contract=contract, seed_root=volume, local_root=local)["status"], "seeded")
            self.assertTrue((cache / "abc123" / "kernel.cubin").is_file())

    def test_failed_publication_preserves_previous_complete_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, contract = Path(temporary), _contract()
            local, volume = root / "local", root / "volume"
            cache = triton_seed.local_cache_dir(contract, local)
            _populate(cache)
            seed = Path(triton_seed.harvest(contract=contract, seed_root=volume, local_root=local)["seed_dir"])
            before = (seed / triton_seed.MANIFEST_NAME).read_bytes()
            (cache / "abc123" / "new.cubin").write_bytes(b"new variant")
            with patch.object(triton_seed, "_write_json", side_effect=OSError("publication interrupted")):
                with self.assertRaises(OSError):
                    triton_seed.package(cache_dir=cache, seed_dir=seed, contract=contract)
            self.assertEqual((seed / triton_seed.MANIFEST_NAME).read_bytes(), before)
            triton_seed.validate(seed, contract=contract, destination=cache)

    def test_concurrent_publishers_leave_one_complete_generation(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as temporary:
            root, contract = Path(temporary), _contract()
            cache, seed = root / "local", root / "seed"
            _populate(cache)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(triton_seed.package, cache_dir=cache, seed_dir=seed, contract=contract) for _ in range(2)]
                for future in futures:
                    future.result()
            triton_seed.validate(seed, contract=contract, destination=cache)

    def test_live_contract_is_none_without_a_gpu_query(self) -> None:
        with patch.object(triton_seed, "_gpu_query", return_value=None):
            self.assertIsNone(triton_seed.live_contract(
                model="100M", architecture="gdn2_hybrid", precision="fp16", microbatch_size=8,
                context_length=2048, gdn_chunk_size=32))


if __name__ == "__main__":
    unittest.main()
