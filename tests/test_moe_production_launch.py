"""Static and CPU qualification for the accepted MoE provider launch path."""

from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import moe_production as production_module
from moe_production import (
    ProductionRequest,
    accepted_identity,
    build_training_command,
    prepare_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


class TestProductionCommand(unittest.TestCase):
    def _request(self, **overrides) -> ProductionRequest:
        values = {
            "run_id": "moe-prod-qualification",
            "dataset_dir": "/data/moe-superbpe",
            "total_steps": 2,
            "precision": "fp16",
            "microbatch_size": 8,
            "source_commit": "a" * 40,
            "checkpoint_every_steps": 1,
        }
        values.update(overrides)
        return ProductionRequest(**values)

    def test_command_freezes_the_accepted_identity(self) -> None:
        command = build_training_command(
            self._request(), run_root=Path("/runs"), schedule=production_module.DISPLAY_SCHEDULE,
        )

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(value("--model-size"), "accepted")
        self.assertEqual(value("--architecture"), "gdn2_hybrid")
        self.assertEqual(value("--gdn-chunk-size"), "32")
        self.assertEqual(value("--load-balancing"), "quantile")
        self.assertEqual(value("--balancing-step-size"), "0")
        self.assertNotIn("substantive", command)

        identity = accepted_identity()
        self.assertEqual(
            (identity["num_experts"], identity["top_k"], identity["expert_d_ff"]),
            (64, 2, 352),
        )
        self.assertEqual(
            (identity["semantic_vocab_size"], identity["padded_vocab_size"]),
            (8_000, 8_192),
        )
        self.assertEqual(identity["load_balancing"], "quantile")

    def test_historical_pilot_launchers_remain_separate(self) -> None:
        for provider, gpu in (("modal", 'gpu="H100"'), ("beam", 'gpu="RTX4090"')):
            source = (ROOT / provider / "moe_production_launch.py").read_text(encoding="utf-8")
            self.assertIn(gpu, source)
            self.assertIn("run_provider_payload", source)
            self.assertNotIn("import moe_pilot", source)


class TestProductionDatasetGate(unittest.TestCase):
    def _dataset(self, root: Path, *, final_token: int) -> Path:
        (root / "train").mkdir(parents=True)
        values = array("H", [0] * 2_048 + [final_token])
        if sys.byteorder != "little":
            values.byteswap()
        payload = values.tobytes()
        shard = root / "train" / "000.bin"
        shard.write_bytes(payload)
        manifest = {
            "schema_version": 2,
            "sequence_format": "context_plus_one",
            "context_length": 2_048,
            "stored_tokens_per_sequence": 2_049,
            "sequences_per_block": 1,
            "shards": [
                {
                    "filename": "train/000.bin",
                    "split": "train",
                    "byte_size": len(payload),
                    "checksum": hashlib.sha256(payload).hexdigest(),
                    "sequence_count": 1,
                    "first_block_id": 0,
                    "last_block_id": 0,
                }
            ],
        }
        (root / "validation").mkdir()
        (root / "validation" / "000.bin").write_bytes(payload)
        manifest["shards"].append({**manifest["shards"][0], "filename": "validation/000.bin",
                                   "split": "validation"})
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # The production gate also reads the corpus run contract for the schedule.
        (root / "run_contract.json").write_text(json.dumps({
            "version": 1, "run_id": "gate", "schema_version": 2, "context_length": 2_048,
            "sequences_per_block": 1, "trainer": {
                "schedule": "wsd", "steps": 2, "warmup_tokens": 2_048, "stable_tokens": 2_048,
                "decay_tokens": 2_048, "minimum_lr_ratio": 0.1, "validation_blocks": 1,
            },
        }), encoding="utf-8")
        return root

    def _request(self, dataset: Path) -> ProductionRequest:
        return ProductionRequest(
            run_id="vocab-gate",
            dataset_dir=str(dataset),
            total_steps=1,
            precision="fp32",
            microbatch_size=1,
            source_commit="b" * 40,
            sequences_per_block=1,
        )

    def test_id_7999_is_a_valid_semantic_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary), final_token=7_999)
            prepared = prepare_dataset(self._request(dataset))
            self.assertEqual(prepared["status"], "ready")
            self.assertEqual(prepared["identity"]["semantic_vocab_size"], 8_000)

    def test_id_8000_is_rejected_even_though_embedding_storage_is_8192(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary), final_token=8_000)
            with self.assertRaisesRegex(ValueError, "outside the semantic vocabulary"):
                prepare_dataset(self._request(dataset))


if __name__ == "__main__":
    unittest.main()


# Import the real entrypoints with only the provider SDK/base launcher replaced.
# No provider installation, credentials, network or GPU is needed.
def _provider_launcher(provider, tmp_path):
    import importlib.abc
    import importlib.machinery
    import importlib.util
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    def decorator(**options):
        def decorate(function):
            function.remote = Mock(name=function.__name__)
            function.options = options
            return function
        return decorate

    base = SimpleNamespace(
        DATA_ROOT=tmp_path / "data", RUN_ROOT=tmp_path / "runs", CACHE_ROOT=tmp_path / "cache",
        REMOTE_REPO=ROOT, DATA_VOLUME=Mock(), RUN_VOLUME=Mock(), CACHE_VOLUME=Mock(),
        TRAINING_SECRET=object(), SECRETS=["existing-secret"], IMAGE=object(), CPU_IMAGE=object(),
        LEGACY_SERVERLESS_IMAGE=object(), RUNTIME_ENV={}, _GPU_FUNCTION_KWARGS={},
        NOOP_VOLUME=Mock(), function=decorator, _repo_root=lambda: ROOT,
        _local_source_commit=lambda: "a" * 40,
        modal=SimpleNamespace(App=lambda *a, **kw: SimpleNamespace(
            function=decorator, local_entrypoint=decorator,
        )),
    )

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__dict__.update(vars(base))

    spec = importlib.util.spec_from_file_location(
        f"test_{provider}_moe_launch", ROOT / provider / "moe_production_launch.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch("importlib.util.spec_from_file_location", return_value=importlib.machinery.ModuleSpec(
        "test_base_launcher", Loader(),
    )):
        spec.loader.exec_module(module)
    return module, base


class TestProviderStreamingEntrypoints(unittest.TestCase):
    def test_both_entrypoints_pass_streaming_flags_and_keep_static_defaults(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as directory:
            for provider in ("modal", "beam"):
                module, base = _provider_launcher(provider, Path(directory))
                for streaming in (False, True):
                    with self.subTest(provider=provider, streaming=streaming):
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            if provider == "modal":
                                kwargs = dict(run_id="test", dataset_dir=str(base.DATA_ROOT / "dataset"),
                                              steps=100, source_commit="a" * 40, dry_run=True)
                                if streaming:
                                    kwargs.update(dataset_shard_bucket="public/corpus",
                                                  dataset_shard_run_id="dataset-001")
                                module.main(**kwargs)
                            else:
                                argv = ["--run-id", "test", "--dataset-dir", str(base.DATA_ROOT / "dataset"),
                                        "--steps", "100", "--source-commit", "a" * 40, "--dry-run"]
                                if streaming:
                                    argv += ["--dataset-shard-bucket", "public/corpus",
                                             "--dataset-shard-run-id", "dataset-001"]
                                self.assertEqual(module.main(argv), 0)
                        payload = json.loads(output.getvalue())
                        self.assertEqual(payload["request"]["dataset_shard_bucket"],
                                         "public/corpus" if streaming else "")
                        self.assertEqual(payload["request"]["dataset_shard_run_id"],
                                         "dataset-001" if streaming else "")
                        self.assertEqual("--dataset-shard-wait-timeout-seconds" in payload["command"], streaming)
                        module.prepare_production_cpu.remote.assert_not_called()

    def test_cpu_preparation_receives_resume_root_and_modal_commits_before_gpu_reload(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            for provider in ("modal", "beam"):
                with self.subTest(provider=provider):
                    module, base = _provider_launcher(provider, Path(directory))
                    request = ProductionRequest(
                        run_id="test", dataset_dir=str(base.DATA_ROOT / "dataset"), total_steps=100,
                        precision="bf16", microbatch_size=1, source_commit="a" * 40,
                        dataset_shard_bucket="public/corpus", dataset_shard_run_id="dataset-001",
                    )
                    events = []
                    base.DATA_VOLUME.reload.side_effect = lambda: events.append("reload")
                    base.DATA_VOLUME.commit.side_effect = lambda: events.append("commit")
                    with patch("moe_production.prepare_dataset", side_effect=lambda *a, **kw: (
                        events.append("prepare") or {"status": "ready"}
                    )) as prepare, patch("moe_production.run_provider_payload", side_effect=lambda *a, **kw: (
                        events.append("train") or {"status": "waiting_for_corpus"}
                    )):
                        payload = production_module.request_payload(request)
                        module.prepare_production_cpu(payload)
                        prepare.assert_called_once_with(request, run_root=base.RUN_ROOT)
                        train = module.train_production_h100 if provider == "modal" else module.train_production_rtx4090
                        self.assertEqual(train(payload)["status"], "waiting_for_corpus")
                    if provider == "modal":
                        self.assertEqual(events, ["reload", "prepare", "commit", "reload", "train"])
                        self.assertIs(module.prepare_production_cpu.options["volumes"][str(base.DATA_ROOT)],
                                      base.DATA_VOLUME)
                        self.assertEqual(module.prepare_production_cpu.options["secrets"], [base.TRAINING_SECRET])
                    else:
                        self.assertEqual(events, ["prepare", "train"])
                        self.assertEqual(module.prepare_production_cpu.options["secrets"], base.SECRETS)
