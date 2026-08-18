"""CPU-only contracts for portable Kaggle Tesla-T4 Triton cache seeding."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
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


def test_t4_cache_contract_matches_deep_decay_execution() -> None:
    module = _load("small_llm_triton_cache_contract_test", KAGGLE / "triton_cache.py")
    contract = module.expected_contract()
    assert contract["gpu_name"] == "Tesla T4"
    assert contract["compute_capability"] == [7, 5]
    assert contract["python"] == "3.13"
    assert contract["torch"] == "2.10.0"
    assert contract["cuda"] == "12.8"
    assert contract["triton"] == "3.6.0"
    assert contract["fla_core"] == "0.5.2"
    assert contract["model"] == "100M"
    assert contract["architecture"] == "gdn2_hybrid"
    assert contract["precision"] == "fp16"
    assert contract["microbatch_size"] == 2
    assert contract["context_length"] == 2048
    assert contract["gdn_chunk_size"] == 32
    assert contract["world_size"] == 2


def test_cache_builder_is_standalone_and_does_not_touch_training_state() -> None:
    source = (KAGGLE / "triton_cache.py").read_text(encoding="utf-8")
    assert "ModelConfig.substantive(" in source
    assert "gdn_chunk_size=GDN_CHUNK_SIZE" in source
    assert "(MICROBATCH_SIZE, CONTEXT_LENGTH)" in source
    assert "objective.backward()" in source
    assert "no optimizer or training state is created" in source
    assert "qualified_runtime_uv_args()" in source
    assert '_uv_worker_command("compile", uv)' in source
    assert '_uv_worker_command("validate", uv)' in source
    assert "dual_t4_train_block64.py" in source


def test_package_and_seed_round_trip_preserves_cache_tree() -> None:
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

        assert result["status"] == "seeded"
        assert (target / "aa" / "kernel.cubin").read_bytes() == b"cubin-bytes"
        assert json.loads((target / "group.json").read_text(encoding="utf-8"))[
            "child_paths"
        ] == [str(target / "aa" / "kernel.cubin")]
        assert (target / module.MANIFEST_NAME).is_file()


def test_stale_kernel_contract_is_rejected_but_cache_remains_optional() -> None:
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

        assert result["status"] == "jit_fallback"
        assert result["rejections"]
        assert target.is_dir()


def test_strict_mode_rejects_an_explicit_invalid_seed() -> None:
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
            try:
                module.prepare_environment(
                    explicit_package=package,
                    strict=True,
                )
            except module.TritonCacheError as error:
                assert "no compatible Triton cache seed" in str(error)
            else:
                raise AssertionError("strict cache seed should fail closed")


def test_block64_preseeds_before_importing_torch() -> None:
    source = (KAGGLE / "dual_t4_train_block64.py").read_text(encoding="utf-8")
    assert "import triton_cache" in source
    prepare = source.index("triton_cache.prepare_environment()")
    torch_import = source.index("    import torch\n")
    assert prepare < torch_import
    assert "filesystem lock" in source


def test_cache_dataset_publication_is_private_by_default() -> None:
    source = (KAGGLE / "triton_cache.py").read_text(encoding="utf-8")
    assert '"datasets",\n            "create"' in source
    assert '"--public"' not in source
    assert '"licenses": [{"name": "other"}]' in source
