"""Real tiny-MoE continuation through the byte-accurate HF bucket fake, no cloud/GPU."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest
import torch

from dataset.src.joint_checkpoint import CheckpointCoordinator
from dataset.src.remote import TwoPhaseCheckpointPublisher, sha256_path
from dataset.src.storage import write_json_atomic
from MOE_model.engine import MoETrainerEngine
from MOE_model.model import MoESmallLLM
from moe_checkpoint_transport import prepare_receipt, receipt_path, restore_checkpoint
from moe_production import ProductionRequest, build_training_command, DISPLAY_SCHEDULE, resolve_resume
from trainer.config import TrainerConfig
from trainer.identity import checkpoint_identity
from trainer.session import TrainingSession
from trainer.shards import SchemaV2ShardReader
from tests.test_hf_bucket_checkpoint import _FakeBucketApi, HuggingFaceBucketCheckpointStore
from tests.test_moe_accepted_geometry import _tiny


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "offline-test")
    monkeypatch.setenv("WANDB_API_KEY", "offline-test")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path / "dataset-a"
    root.mkdir()
    rows = torch.arange(4 * 4 * 9).remainder(32).to(torch.uint16).numpy().tobytes()
    shard = root / "train.bin"
    shard.write_bytes(rows)
    manifest = {"schema_version": 2, "sequence_format": "context_plus_one",
                "context_length": 8, "stored_tokens_per_sequence": 9, "sequences_per_block": 4,
                "shards": [{"filename": "train.bin", "split": "train", "sequence_count": 16,
                            "byte_size": len(rows), "checksum": sha256_path(shard),
                            "first_block_id": 0, "last_block_id": 3}]}
    validation = root / "validation.bin"
    validation.write_bytes(torch.arange(4 * 9).add(7).remainder(32).to(torch.uint16).numpy().tobytes())
    manifest["shards"].append({"filename": "validation.bin", "split": "validation", "sequence_count": 4,
                              "byte_size": validation.stat().st_size, "checksum": sha256_path(validation),
                              "first_block_id": 0, "last_block_id": 0})
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "run_contract.json", {"fixed": "corpus identity", "trainer": {
        "schedule": "wsd", "steps": 4, "warmup_tokens": 32, "stable_tokens": 64,
        "decay_tokens": 32, "minimum_lr_ratio": .1, "validation_blocks": 1}})
    request = ProductionRequest(run_id="portable", dataset_dir=str(root), total_steps=4,
        precision="fp32", microbatch_size=1, source_commit="a" * 40,
        checkpoint_bucket="owner/checkpoints", wandb_entity="owner", resume="new")
    model_config = replace(_tiny(), router_scoring="sqrt_softplus", router_z_loss_coefficient=0.,
                           load_balancing="quantile")
    config = TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32", microbatch_size=1,
                           schedule="wsd", warmup_tokens=32, stable_tokens=64, decay_tokens=32,
                           checkpoint_every_steps=1, evaluation_every_steps=1)
    api = _FakeBucketApi()
    store = HuggingFaceBucketCheckpointStore("owner/checkpoints", api=api)
    with patch("moe_checkpoint_transport.HuggingFaceBucketCheckpointStore", return_value=store):
        yield root, request, model_config, config, store, api
    torch.set_num_threads(previous_threads)


def _session(root, model_config, config, checkpoint_root):
    engine = MoETrainerEngine(MoESmallLLM(model_config), config, device="cpu")
    reader = SchemaV2ShardReader(root, semantic_vocab_size=32)
    session = TrainingSession(engine, reader)
    identity = checkpoint_identity(root, model_config=model_config, trainer_config=config)
    coordinator = CheckpointCoordinator(checkpoint_root, configuration_hash=identity[0],
                                        source_hash=identity[1], schema_hash=identity[2])
    return session, coordinator


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("source_upgrade", [False, True])
def test_real_checkpoint_crosses_empty_provider_with_new_microbatch(tmp_path, corpus, source_upgrade):
    root, request, model_config, config, store, _ = corpus
    a, b = tmp_path / "provider-a", tmp_path / "provider-b"
    assert restore_checkpoint(request, run_root=a) is None
    prepare_receipt(request, run_root=a, previous=None)
    session, coordinator = _session(root, model_config, config, a / request.run_id / "checkpoints")
    session.step()
    session.step()
    checkpoint_id = "step-00000002"
    session.save_checkpoint(coordinator, checkpoint_id)
    before = session.engine.state_dict()
    publisher = TwoPhaseCheckpointPublisher(store, run_id=request.run_id)
    coordinator.publish(publisher, checkpoint_id=checkpoint_id,
                        drive_manifest=json.loads(receipt_path(request, a).read_text()))
    # Entirely separate paths emulate a new account, with only the dataset reproducible.
    new_data = tmp_path / "dataset-b"
    shutil.copytree(root, new_data)
    migrated_request = replace(request, dataset_dir=str(new_data), resume="latest", microbatch_size=2)
    if source_upgrade:
        migrated_request = replace(migrated_request, source_commit="b" * 40,
                                   resume_source_commit=request.source_commit,
                                   validation_microbatch_size=4, async_checkpoint_upload=True)
    previous = restore_checkpoint(migrated_request, run_root=b)
    prepare_receipt(migrated_request, run_root=b, previous=previous)
    receipt = json.loads(receipt_path(migrated_request, b).read_text())
    if source_upgrade:
        # Old executors require this exact v1 receipt, including the original commit.
        assert receipt == previous
        assert receipt["source_commit"] == request.source_commit
        prepare_receipt(migrated_request, run_root=b, previous=receipt)
        assert receipt == json.loads(receipt_path(migrated_request, b).read_text())
    assert resolve_resume(migrated_request, run_root=b)["remaining_steps"] == 2
    target_config = replace(config, microbatch_size=2, checkpoint_every_steps=2, evaluation_every_steps=2)
    restored, target_coordinator = _session(new_data, model_config, target_config,
                                          b / request.run_id / "checkpoints")
    # Loading uses the snapshot identity, exactly as the real MoE setup does.
    target_coordinator.configuration_hash = coordinator.configuration_hash
    restored.load_checkpoint(target_coordinator, checkpoint_id)
    assert restored.source.last_acknowledged_block_id == 1
    after = restored.engine.state_dict()
    for key in ("model", "optimizer", "scaler", "consumed_tokens", "global_step",
                "python_rng_state", "torch_rng_state", "cuda_rng_states"):
        _tree_equal(before[key], after[key])
    assert before["scheduler"]["committed_tokens"] == after["scheduler"]["committed_tokens"] == 64
    assert before["scheduler"]["last_lr"] == after["scheduler"]["last_lr"]
    continuous_metrics = session.step()
    resumed_metrics = restored.step()
    assert resumed_metrics.step == 3
    assert restored.source.last_acknowledged_block_id == 2
    assert restored.engine.consumed_tokens == 96
    if source_upgrade:
        target_coordinator.executor_metadata = {
            "version": 1, "source_commit": migrated_request.source_commit,
            "run_source_commit": request.source_commit,
        }
        saved = restored.save_checkpoint(target_coordinator, "step-00000003")
        metadata = json.loads((saved / "checkpoint.json").read_text())
        assert metadata["executor"]["source_commit"] == "b" * 40
        assert metadata["executor"]["run_source_commit"] == "a" * 40
    assert resumed_metrics.loss == pytest.approx(continuous_metrics.loss, abs=2e-5)
    for key, value in session.engine.model.state_dict().items():
        torch.testing.assert_close(value, restored.engine.model.state_dict()[key], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("change", [{"learning_rate": 1e-3}, {"precision": "bf16"},
                                   {"warmup_tokens": 64}, {"muon_momentum": .8}])
def test_execution_migration_refuses_scientific_changes(tmp_path, corpus, change):
    root, _, model_config, config, _, _ = corpus
    source, _ = _session(root, model_config, config, tmp_path / "source")
    source.step()
    target, _ = _session(root, model_config, replace(config, **change), tmp_path / "target")
    original = target.engine.state_dict()
    with pytest.raises(ValueError, match="scientific configuration"):
        target.engine.load_state_dict(source.engine.state_dict())
    _tree_equal(original["model"], target.engine.state_dict()["model"])


def test_empty_account_never_silently_restarts_and_receipt_binds_corpus(tmp_path, corpus):
    root, request, _, _, _, _ = corpus
    with pytest.raises(ValueError, match="--resume new"):
        restore_checkpoint(replace(request, resume="latest"), run_root=tmp_path / "empty")
    prepare_receipt(request, run_root=tmp_path, previous=None)
    receipt = json.loads(receipt_path(request, tmp_path).read_text())
    for changed in (replace(request, source_commit="b" * 40), replace(request, wandb_entity="other")):
        with pytest.raises(ValueError, match="identity mismatch"):
            prepare_receipt(changed, run_root=tmp_path, previous=receipt)
    contract = json.loads((root / "run_contract.json").read_text())
    write_json_atomic(root / "run_contract.json", {**contract, "changed": True})
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_receipt(request, run_root=tmp_path, previous=receipt)


def test_receipt_uses_existing_geometry_only_schedule_fallback(tmp_path, corpus):
    root, request, _, _, _, _ = corpus
    write_json_atomic(root / "run_contract.json", {"planned_train_blocks": 100,
        "context_length": 8, "sequences_per_block": 4, "validation_blocks": 1})
    prepare_receipt(request, run_root=tmp_path, previous=None)
    assert json.loads(receipt_path(request, tmp_path).read_text())["validation_blocks"] == 1


def test_failed_upload_preserves_last_pointer_and_corrupt_restore_fails(tmp_path, corpus):
    root, request, model_config, config, store, api = corpus
    prepare_receipt(request, run_root=tmp_path, previous=None)
    session, coordinator = _session(root, model_config, config, tmp_path / "portable/checkpoints")
    publisher = TwoPhaseCheckpointPublisher(store, run_id=request.run_id)
    receipt = json.loads(receipt_path(request, tmp_path).read_text())
    for step in (1, 2):
        session.step()
        session.save_checkpoint(coordinator, f"step-{step:08d}")
    coordinator.publish(publisher, checkpoint_id="step-00000001", drive_manifest=receipt)
    pointer_before = store.read_json("run/portable/latest.json")
    with patch.object(store, "upload_tree", side_effect=RuntimeError("provider killed during upload")):
        with pytest.raises(RuntimeError, match="killed"):
            coordinator.publish(publisher, checkpoint_id="step-00000002", drive_manifest=receipt)
    assert store.read_json("run/portable/latest.json") == pointer_before
    api.objects["run/portable/checkpoints/step-00000001/last/trainer_state.pkl"] = b"corrupt"
    with pytest.raises(RuntimeError):
        restore_checkpoint(replace(request, resume="latest"), run_root=tmp_path / "new-account")
    assert not (tmp_path / "new-account/portable/checkpoints/step-00000001").exists()


def test_restore_honors_beam_distributed_volume_fsync_policy(tmp_path, corpus, monkeypatch):
    root, request, model_config, config, store, _ = corpus
    prepare_receipt(request, run_root=tmp_path, previous=None)
    session, coordinator = _session(root, model_config, config, tmp_path / "portable/checkpoints")
    session.step()
    session.save_checkpoint(coordinator, "step-00000001")
    coordinator.publish(TwoPhaseCheckpointPublisher(store, run_id=request.run_id),
                        checkpoint_id="step-00000001",
                        drive_manifest=json.loads(receipt_path(request, tmp_path).read_text()))
    monkeypatch.setenv("SMALL_LLM_CHECKPOINT_FSYNC", "0")
    with patch("dataset.src.joint_checkpoint.os.fsync", side_effect=AssertionError("Beam fsync")):
        restore_checkpoint(replace(request, resume="latest"), run_root=tmp_path / "beam")


@pytest.mark.parametrize("best_tree", ["last", "best"])
def test_bucket_retains_latest_and_best_but_prunes_other_steps(corpus, best_tree):
    _, _, _, _, store, api = corpus
    for step in (1, 2, 3):
        api.objects[f"run/portable/checkpoints/step-{step:08d}/last/state"] = b"weights"
    api.objects[f"run/portable/checkpoints/step-00000001/{best_tree}/state"] = b"weights"
    store.write_json("run/portable/latest.json", {"checkpoint_id": "step-00000003"})
    store.write_json("run/portable/best.json", {"checkpoint_id": "step-00000001",
                     "best_prefix": f"run/portable/checkpoints/step-00000001/{best_tree}", "metric": -2.})
    store.prune_run_checkpoints(run_id="portable", checkpoint_id="step-00000003", keep_best=True)
    assert f"run/portable/checkpoints/step-00000001/{best_tree}/state" in api.objects
    assert "run/portable/checkpoints/step-00000002/last/state" not in api.objects
    assert "run/portable/checkpoints/step-00000003/last/state" in api.objects
    assert store.read_json("run/portable/best.json")["metric"] == -2.


def test_local_volume_missing_publication_sidecars_recovers_verified_metadata(tmp_path, corpus):
    root, request, model_config, config, store, _ = corpus
    prepare_receipt(request, run_root=tmp_path, previous=None)
    session, coordinator = _session(root, model_config, config, tmp_path / "portable/checkpoints")
    session.step()
    session.save_checkpoint(coordinator, "step-00000001")
    checkpoint = tmp_path / "portable/checkpoints/step-00000001"
    original = (checkpoint / "trainer_state.pkl").read_bytes()
    coordinator.publish(TwoPhaseCheckpointPublisher(store, run_id=request.run_id),
                        checkpoint_id="step-00000001",
                        drive_manifest=json.loads(receipt_path(request, tmp_path).read_text()))
    for name in ("drive_manifest.json", "checkpoint_manifest.json"):
        (checkpoint / name).unlink()
    receipt = restore_checkpoint(replace(request, resume="latest"), run_root=tmp_path)
    assert receipt["run_id"] == "portable"
    assert (checkpoint / "trainer_state.pkl").read_bytes() == original
    assert (checkpoint / "checkpoint_manifest.json").is_file()


@pytest.mark.parametrize("latest_uploaded", [False, True])
def test_completed_run_repairs_final_upload_and_best_without_gpu(tmp_path, corpus, latest_uploaded):
    from moe_checkpoint_transport import finalize_completed_run
    root, request, model_config, config, store, _ = corpus
    request = replace(request, total_steps=1)
    prepare_receipt(request, run_root=tmp_path, previous=None)
    session, coordinator = _session(root, model_config, config, tmp_path / "portable/checkpoints")
    session.step()
    session.save_checkpoint(coordinator, "step-00000001", validation_metrics={"loss": 2.})
    if latest_uploaded:
        coordinator.publish(TwoPhaseCheckpointPublisher(store, run_id=request.run_id),
                            checkpoint_id="step-00000001",
                            drive_manifest=json.loads(receipt_path(request, tmp_path).read_text()))
    finalize_completed_run(request, run_root=tmp_path)
    assert store.read_json("run/portable/latest.json")["checkpoint_id"] == "step-00000001"
    assert store.read_json("run/portable/best.json")["metric"] == -2.
    with patch.object(store, "upload_tree", side_effect=AssertionError("duplicate upload")):
        finalize_completed_run(request, run_root=tmp_path)


def test_both_gpu_recipes_publish_each_checkpoint_to_same_wandb_identity(tmp_path, corpus):
    _, request, _, _, _, _ = corpus
    prepare_receipt(request, run_root=tmp_path, previous=None)
    for microbatch in (16, 64):
        command = build_training_command(replace(request, microbatch_size=microbatch),
                                         run_root=tmp_path, schedule=DISPLAY_SCHEDULE)
        value = lambda flag: command[command.index(flag) + 1]
        assert value("--remote-publish-every-steps") == value("--checkpoint-every-steps")
        assert value("--wandb-run-id") == "portable"
        assert value("--wandb-entity") == "owner"
        assert value("--wandb-resume") == "allow"
        assert "--remote-keep-latest-and-best" in command


def test_real_cli_publishes_and_continues_wandb_on_a_new_account(tmp_path, corpus):
    from MOE_model.__main__ import main
    from tests.test_wandb_logging import _Wandb
    from trainer.state import load_trainer_state_file
    root, request, model_config, _, store, _ = corpus
    schedule = {"steps": 4, "warmup_tokens": 32, "stable_tokens": 64,
                "decay_tokens": 32, "minimum_lr_ratio": .1, "validation_blocks": 1}
    wandb = _Wandb()
    a, b = tmp_path / "account-a", tmp_path / "account-b"
    first = replace(request, total_steps=2, checkpoint_every_steps=1)
    prepare_receipt(first, run_root=a, previous=None)
    with patch("MOE_model.setup._model_config_from_args", return_value=model_config), patch(
        "trainer.remote_publication.HuggingFaceBucketCheckpointStore", return_value=store,
    ), patch.dict(sys.modules, {"wandb": wandb}), patch(
        "trainer.observation._source_identity", return_value={"source_commit": "a" * 40},
    ):
        command = build_training_command(first, run_root=a, schedule=schedule)
        assert main(command[3:] + ["--device", "cpu"]) == 0
        assert store.read_json("run/portable/latest.json")["checkpoint_id"] == "step-00000002"
        assert store.read_json("run/portable/best.json") is not None
        assert [row["checkpoint/remote_id"] for row in wandb.run.logged
                if "checkpoint/remote_id" in row] == ["step-00000001", "step-00000002"]
        first_identity = {key: wandb.init_kwargs[key] for key in ("id", "entity", "project")}
        # Simulate a kill after latest.json succeeded but before best.json did.
        store.write_json("run/portable/best.json", {**store.read_json("run/portable/best.json"), "metric": -100.})
        # A different GPU on the same account must also accept its existing
        # observation directory, not just a newly empty provider volume.
        same_account = replace(request, total_steps=3, resume="latest",
                               microbatch_size=2, checkpoint_every_steps=2)
        previous = restore_checkpoint(same_account, run_root=a)
        prepare_receipt(same_account, run_root=a, previous=previous)
        command = build_training_command(same_account, run_root=a, schedule=schedule)
        assert main(command[3:] + ["--device", "cpu"]) == 0
        assert store.read_json("run/portable/best.json")["metric"] > -100.
        new_root = tmp_path / "restaged-corpus"
        shutil.copytree(root, new_root)
        second = replace(request, dataset_dir=str(new_root), resume="latest",
                         microbatch_size=4, checkpoint_every_steps=1)
        previous = restore_checkpoint(second, run_root=b)
        prepare_receipt(second, run_root=b, previous=previous)
        command = build_training_command(second, run_root=b, schedule=schedule)
        assert main(command[3:] + ["--device", "cpu"]) == 0
        assert wandb.init_kwargs["resume"] == "must"
        assert first_identity == {key: wandb.init_kwargs[key] for key in ("id", "entity", "project")}
        train_steps = [row["trainer/global_step"] for row in wandb.run.logged if "train/loss" in row]
        assert train_steps == [1, 2, 3, 4]
        assert store.read_json("run/portable/latest.json")["checkpoint_id"] == "step-00000004"
        state = load_trainer_state_file(b / "portable/checkpoints/step-00000004/trainer_state.pkl")
        assert state["global_step"] == 4 and state["consumed_tokens"] == 128
        assert state["config"]["microbatch_size"] == 4


@pytest.mark.parametrize("changes", [
    {"wandb_entity": "other"}, {"checkpoint_bucket": "other/bucket"},
    {"resume_source_commit": "c" * 40}, {"validation_blocks": 2},
])
def test_source_migration_never_authorizes_other_identity_drift(tmp_path, corpus, changes):
    _, request, _, _, _, _ = corpus
    prepare_receipt(request, run_root=tmp_path, previous=None)
    receipt = json.loads(receipt_path(request, tmp_path).read_text())
    migrated = replace(request, resume="latest", source_commit="b" * 40,
                       resume_source_commit="a" * 40)
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_receipt(replace(migrated, **changes), run_root=tmp_path, previous=receipt)
    assert json.loads(receipt_path(request, tmp_path).read_text()) == receipt


def test_source_migration_requires_complete_checkpoint(tmp_path, corpus):
    _, request, _, _, _, _ = corpus
    prepare_receipt(request, run_root=tmp_path, previous=None)
    receipt = json.loads(receipt_path(request, tmp_path).read_text())
    migrated = replace(request, resume="latest", source_commit="b" * 40,
                       resume_source_commit="a" * 40)
    with pytest.raises(ValueError, match="complete resume checkpoint"):
        prepare_receipt(migrated, run_root=tmp_path, previous=receipt)


def test_previous_executor_resumes_new_checkpoint(tmp_path, corpus):
    import os
    from dataclasses import asdict
    import pickle
    import subprocess
    old_checkout = os.environ.get("SMALL_LLM_ROLLBACK_CHECKOUT")
    if not old_checkout:
        pytest.skip("set SMALL_LLM_ROLLBACK_CHECKOUT for cross-executor rollback proof")
    root, request, model_config, config, _, _ = corpus
    run_root = tmp_path / "rollback"
    prepare_receipt(request, run_root=run_root, previous=None)
    receipt = json.loads(receipt_path(request, run_root).read_text())
    checkpoint_root = run_root / request.run_id / "checkpoints"
    session, coordinator = _session(root, model_config, config, checkpoint_root)
    session.step()
    coordinator.executor_metadata = {"version": 1, "source_commit": "b" * 40,
                                     "run_source_commit": request.source_commit}
    session.save_checkpoint(coordinator, "step-00000001")
    # The previous executable receives the original receipt and unchanged tensor format.
    setup_file = tmp_path / "rollback-input.pkl"
    result_file = tmp_path / "rollback-output.pkl"
    setup_file.write_bytes(pickle.dumps((root, model_config, config, checkpoint_root,
                                        asdict(replace(request, resume="latest")), run_root, receipt)))
    program = """
import dataclasses, pickle, sys
from tests.test_moe_provider_continuation import _session
from moe_checkpoint_transport import prepare_receipt
from moe_production import ProductionRequest
root, model, config, checkpoint_root, request, run_root, receipt = pickle.load(open(sys.argv[1], 'rb'))
# Reconstruct the original request using the previous executable field set.
request = ProductionRequest(**{field.name: request[field.name] for field in dataclasses.fields(ProductionRequest)})
prepare_receipt(request, run_root=run_root, previous=receipt)
session, coordinator = _session(root, model, config, checkpoint_root)
session.load_checkpoint(coordinator, 'step-00000001')
session.step()
pickle.dump(session.engine.state_dict(), open(sys.argv[2], 'wb'))
"""
    env = {**os.environ, "PYTHONPATH": old_checkout, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "-c", program, str(setup_file), str(result_file)],
                   cwd=old_checkout, env=env, check=True, capture_output=True, text=True)
    session.step()
    old_state = pickle.loads(result_file.read_bytes())
    state = session.engine.state_dict()
    for key in ("model", "optimizer", "scheduler", "scaler", "consumed_tokens", "global_step"):
        _tree_equal(state[key], old_state[key])
