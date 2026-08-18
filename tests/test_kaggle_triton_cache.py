"""CPU-only contracts for portable Kaggle Tesla-T4 Triton cache seeding."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamp(module, root: Path) -> None:
    module._write_json(
        root / module.BUILD_STAMP_NAME,
        {
            "schema_version": module.SCHEMA_VERSION,
            "cache_id": module.CACHE_ID,
            "phase": "validate",
            "contract": module.expected_contract(),
            "kernel_contract_sha256": module.kernel_contract_sha256(),
        },
    )


class KaggleTritonCacheTests(unittest.TestCase):
    def test_t4_cache_contract_matches_deep_decay_execution(self) -> None:
        module = _load("small_llm_triton_cache_contract_test", KAGGLE / "triton_cache.py")
        contract = module.expected_contract()
        self.assertEqual(contract["gpu_name"], "Tesla T4")
        self.assertEqual(contract["compute_capability"], [7, 5])
        self.assertEqual(contract["python"], "3.13")
        self.assertEqual(contract["torch"], "2.10.0")
        self.assertEqual(contract["cuda"], "12.8")
        self.assertEqual(contract["triton"], "3.6.0")
        self.assertEqual(contract["fla_core"], "0.5.2")
        self.assertEqual(contract["model"], "100M")
        self.assertEqual(contract["architecture"], "gdn2_hybrid")
        self.assertEqual(contract["precision"], "fp16")
        self.assertEqual(contract["microbatch_size"], 2)
        self.assertEqual(contract["context_length"], 2048)
        self.assertEqual(contract["gdn_chunk_size"], 32)
        self.assertEqual(contract["world_size"], 2)

    def test_cache_builder_is_standalone_and_does_not_touch_training_state(self) -> None:
        source = (KAGGLE / "triton_cache.py").read_text(encoding="utf-8")
        self.assertIn("ModelConfig.substantive(", source)
        self.assertIn("gdn_chunk_size=GDN_CHUNK_SIZE", source)
        self.assertIn("(MICROBATCH_SIZE, CONTEXT_LENGTH)", source)
        self.assertIn("objective.backward()", source)
        self.assertIn("no optimizer or training state is created", source)
        self.assertIn("qualified_runtime_uv_args()", source)
        self.assertIn('_uv_worker_command("compile", uv)', source)
        self.assertIn('_uv_worker_command("validate", uv)', source)
        self.assertIn("dual_t4_train_block64.py", source)

    def test_package_and_seed_round_trip_preserves_cache_tree(self) -> None:
        module = _load("small_llm_triton_cache_roundtrip_test", KAGGLE / "triton_cache.py")
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            built = temp / "built"
            target = temp / "canonical"
            package = temp / "package"
            built.mkdir()
            (built / "aa").mkdir()
            (built / "aa" / "kernel.cubin").write_bytes(b"cubin-bytes")
            (built / "group.json").write_text(
                json.dumps({"child_paths": [str(target / "aa" / "kernel.cubin")]}),
                encoding="utf-8",
            )
            _stamp(module, built)

            with mock.patch.dict(
                os.environ,
                {
                    module.CACHE_DIR_ENV: str(target),
                    module.DATASET_DIR_ENV: str(package),
                },
                clear=False,
            ):
                module.package_cache(cache_root=built, output_dir=package)
                result = module.prepare_environment(strict=True)

            self.assertEqual(result["status"], "seeded")
            self.assertEqual(
                (target / "aa" / "kernel.cubin").read_bytes(),
                b"cubin-bytes",
            )
            self.assertEqual(
                json.loads((target / "group.json").read_text(encoding="utf-8"))[
                    "child_paths"
                ],
                [str(target / "aa" / "kernel.cubin")],
            )
            self.assertTrue((target / module.MANIFEST_NAME).is_file())

    def test_stale_kernel_contract_is_rejected_but_cache_remains_optional(self) -> None:
        module = _load("small_llm_triton_cache_stale_test", KAGGLE / "triton_cache.py")
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            built = temp / "built"
            target = temp / "canonical"
            package = temp / "package"
            built.mkdir()
            (built / "kernel.cubin").write_bytes(b"cache")
            _stamp(module, built)

            with mock.patch.dict(
                os.environ,
                {module.CACHE_DIR_ENV: str(target)},
                clear=False,
            ):
                module.package_cache(cache_root=built, output_dir=package)
                manifest_path = package / module.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["kernel_contract_sha256"] = "0" * 64
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                result = module.prepare_environment(
                    explicit_package=package,
                    strict=False,
                )

            self.assertEqual(result["status"], "jit_fallback")
            self.assertTrue(result["rejections"])
            self.assertTrue(target.is_dir())

    def test_strict_mode_rejects_an_explicit_invalid_seed(self) -> None:
        module = _load("small_llm_triton_cache_strict_test", KAGGLE / "triton_cache.py")
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            package = temp / "broken"
            target = temp / "canonical"
            package.mkdir()
            (package / module.MANIFEST_NAME).write_text("{}\n", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {module.CACHE_DIR_ENV: str(target)},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    module.TritonCacheError,
                    "no compatible Triton cache seed",
                ):
                    module.prepare_environment(
                        explicit_package=package,
                        strict=True,
                    )

    def test_block64_preseeds_before_importing_torch(self) -> None:
        source = (KAGGLE / "dual_t4_train_block64.py").read_text(encoding="utf-8")
        self.assertIn("import triton_cache", source)
        prepare = source.index("triton_cache.prepare_environment()")
        torch_import = source.index("    import torch\n")
        self.assertLess(prepare, torch_import)
        self.assertIn("filesystem lock", source)

    def test_cache_dataset_publication_is_private_by_default(self) -> None:
        source = (KAGGLE / "triton_cache.py").read_text(encoding="utf-8")
        self.assertIn('"datasets",\n            "create"', source)
        self.assertNotIn('"--public"', source)
        self.assertIn('"licenses": [{"name": "other"}]', source)


if __name__ == "__main__":
    unittest.main()
