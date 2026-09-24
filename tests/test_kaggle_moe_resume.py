"""The Kaggle run-003 entrypoint must fail closed before touching the live run."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

import torch
from trainer.observation import RunObservation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("small_llm_kaggle_moe_resume", ROOT / "kaggle" / "moe_resume.py")
assert SPEC and SPEC.loader
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class KaggleMoEResumeTests(TestCase):
    def test_run_is_pinned_to_actual_003_bucket_and_origin(self) -> None:
        req = launch.request("a" * 40, Path("/tmp/moe"))
        self.assertEqual(req.dataset_shard_bucket, "abcastor/small-llm-corpus-100b-v2-dataset")
        self.assertEqual(req.dataset_shard_run_id, "moe-100b-superbpe-b64-dataset-003")
        self.assertEqual(req.checkpoint_bucket, "roccoangelella/small-llm-100m-qualification-checkpoints")
        self.assertEqual(req.resume_source_commit, "5f08941028cce913f336a8aa2f8ce5fb6231ca4c")
        self.assertEqual(req.resume, "latest")
        self.assertEqual(req.precision, "bf16")
        self.assertEqual(req.total_steps, 762940)

    def test_wrong_checkout_or_public_head_fails(self) -> None:
        with mock.patch.object(launch, "_git", side_effect=["main"]):
            with self.assertRaisesRegex(RuntimeError, "branch"):
                launch.clean_branch_commit()
        with mock.patch.object(launch, "_git", side_effect=[launch.BRANCH, "", "a" * 40]), \
             mock.patch.object(launch.subprocess, "check_output", return_value="b" * 40 + "\trefs/heads/moe-8e-top1"):
            with self.assertRaisesRegex(RuntimeError, "public"):
                launch.clean_branch_commit()

    def test_remote_latest_checks_run_receipt_before_restore(self) -> None:
        expected = {"version": 1, "run_id": launch.RUN_ID, "source_commit": launch.RUN_ORIGIN,
                    "checkpoint_bucket": launch.CHECKPOINT_BUCKET,
                    "wandb": {"entity": launch.WANDB_ENTITY, "project": "Small-LLM", "id": launch.RUN_ID}}
        prefix = f"run/{launch.RUN_ID}/checkpoints/step-00265000/last"
        pointer = {"checkpoint_id": "step-00265000", "last_prefix": prefix}
        store = mock.Mock()
        store.read_json.side_effect = [pointer, expected]
        with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}), \
             mock.patch("dataset.src.hf_bucket_checkpoint.HuggingFaceBucketCheckpointStore", return_value=store):
            self.assertEqual(launch.remote_latest(launch.request("a" * 40, Path("/tmp/moe")))[0],
                             "step-00265000")
        store.read_json.side_effect = [pointer, {**expected, "source_commit": "b" * 40}]
        with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}), \
             mock.patch("dataset.src.hf_bucket_checkpoint.HuggingFaceBucketCheckpointStore", return_value=store):
            with self.assertRaisesRegex(RuntimeError, "receipt"):
                launch.remote_latest(launch.request("a" * 40, Path("/tmp/moe")))

    def test_running_writer_cannot_start_training(self) -> None:
        env = {"HF_TOKEN": "fake", "WANDB_API_KEY": "fake",
               "SMALL_LLM_HF_REPO_ID": "roccoangelella/small-llm-100m-qualification"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env), \
             mock.patch.object(launch, "clean_branch_commit", return_value="a" * 40), \
             mock.patch.object(launch, "remote_latest", return_value=("step-00265000", {"dataset_manifest_sha256": "sha"})), \
             mock.patch.object(launch, "wandb_state", return_value="running"), \
             mock.patch.object(launch, "prepare_dataset") as stage, \
             mock.patch.object(launch, "run_provider_payload") as train:
            with self.assertRaisesRegex(RuntimeError, "previous writer"):
                launch.main(["--work-dir", tmp, "--start"])
            stage.assert_not_called()
            train.assert_not_called()

    def test_preflight_is_cpu_only_and_detects_pointer_movement(self) -> None:
        env = {"HF_TOKEN": "fake", "WANDB_API_KEY": "fake",
               "SMALL_LLM_HF_REPO_ID": "roccoangelella/small-llm-100m-qualification"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env), \
             mock.patch.object(launch, "clean_branch_commit", return_value="a" * 40), \
             mock.patch.object(launch, "remote_latest", side_effect=[
                 ("step-00265000", {"dataset_manifest_sha256": "sha"}),
                 ("step-00266000", {"dataset_manifest_sha256": "sha"}),
             ]), mock.patch.object(launch, "wandb_state", return_value="running"), \
             mock.patch.object(launch, "prepare_dataset", return_value={"status": "ready", "identity": launch.accepted_identity()}), \
             mock.patch.object(launch, "run_provider_payload") as train:
            with self.assertRaisesRegex(RuntimeError, "advanced"):
                launch.main(["--work-dir", tmp])
            train.assert_not_called()

    def test_same_executor_can_resume_with_older_run_origin(self) -> None:
        executor, origin = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text("{}")
            args = SimpleNamespace(
                experiment_dir=root / "artifacts", dataset_dir=root, dataset_manifest=None,
                source_commit=executor, run_source_commit=origin, resume="step-00000001",
                initialization="normal", sequences_per_block=None, probe_sequences=0,
                profile_at_steps=[], checkpoint_at_steps=[], probe_lm_logits=False,
            )
            model = torch.nn.Linear(2, 2)
            engine = SimpleNamespace(model=model, optimizer=SimpleNamespace(param_groups=[{"params": list(model.parameters())}]),
                                     global_step=1, consumed_tokens=64, device=torch.device("cpu"))
            config = SimpleNamespace(version=3, as_dict=lambda: {"version": 3})
            trainer = SimpleNamespace(as_dict=lambda: {"precision": "bf16"})
            source = {"source_commit": executor, "source_tree_sha256": "tree", "source_tree_dirty": False}
            with mock.patch("trainer.observation._source_identity", return_value=source):
                RunObservation(args, engine, config, trainer, lambda *_: None)
                RunObservation(args, engine, config, trainer, lambda *_: None)
                args.source_commit = "c" * 40
                with self.assertRaisesRegex(ValueError, "observation origin"):
                    RunObservation(args, engine, config, trainer, lambda *_: None)

    def test_training_entrypoint_requires_canonical_command(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            launch.kaggle_popen([sys.executable, "-m", "trainer"])
        with mock.patch.object(launch.subprocess, "Popen") as popen:
            launch.kaggle_popen([sys.executable, "-m", "MOE_model", "--resume", "step-00000001"], cwd="/tmp")
            command = popen.call_args.args[0]
            self.assertEqual(Path(command[1]).name, "moe_train_single_t4.py")
            self.assertEqual(command[2:], ["--resume", "step-00000001"])
