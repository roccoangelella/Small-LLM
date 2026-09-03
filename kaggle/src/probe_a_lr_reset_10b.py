#!/usr/bin/env python3
"""Safe entrypoint for disposable Probe A LR-reset branches on Kaggle.

The implementation lives in ``probe_a_lr_reset_10b_impl.py``. This thin shim
exists because the shared deep-decay HF runtime helper re-execs its own file;
Probe A must instead restart back into this probe entrypoint before delegating.

Probe A is intentionally fixed to start from the current available strict-best
checkpoint ``step-00071750`` rather than from the rolling latest checkpoint.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

KAGGLE = Path(__file__).resolve().parent
ROOT = KAGGLE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(KAGGLE) not in sys.path:
    sys.path.insert(0, str(KAGGLE))

import deep_decay_10b_from_15500 as deep_decay

HF_HUB_VERSION = getattr(deep_decay, "HF_HUB_VERSION", "1.5.0")
PROBE_BASE_HF_REPO_ID = (
    os.environ.get(
        "SMALL_LLM_PROBE_A_BASE_REPO_ID",
        "roccoangelella/small-llm-100m-qualification",
    ).strip()
    or "roccoangelella/small-llm-100m-qualification"
)
PROBE_SOURCE_CHECKPOINT_ID = "step-00071750"
PROBE_SOURCE_KIND = "fixed_best_model_checkpoint"
_POST_SAVE_METADATA = frozenset(
    {"local_manifest.json", "drive_manifest.json", "checkpoint_manifest.json"}
)


def _force_probe_hf_identity() -> None:
    """Force Probe A to the 100M qualification HF namespace.

    Kaggle notebooks can retain environment variables from older 20M runs. Probe A
    is a 100M/10B experiment, so all implicit runtime helpers must resolve the
    100M model repo, checkpoint bucket, and dataset bucket unless a dedicated
    Probe-A override is supplied.
    """

    desired = {
        "SMALL_LLM_HF_REPO_ID": PROBE_BASE_HF_REPO_ID,
        "SMALL_LLM_HF_CHECKPOINT_BUCKET_ID": os.environ.get(
            "SMALL_LLM_PROBE_A_CHECKPOINT_BUCKET_ID",
            f"{PROBE_BASE_HF_REPO_ID}-checkpoints",
        ).strip()
        or f"{PROBE_BASE_HF_REPO_ID}-checkpoints",
        "SMALL_LLM_HF_DATASET_BUCKET_ID": os.environ.get(
            "SMALL_LLM_PROBE_A_DATASET_BUCKET_ID",
            f"{PROBE_BASE_HF_REPO_ID}-datasets",
        ).strip()
        or f"{PROBE_BASE_HF_REPO_ID}-datasets",
    }
    previous = {key: os.environ.get(key) for key in desired}
    for key, value in desired.items():
        os.environ[key] = value

    overridden = {
        key: {"previous": value, "current": desired[key]}
        for key, value in previous.items()
        if value and value != desired[key]
    }
    if overridden:
        print(
            json.dumps({"probe_a_hf_identity_override": overridden}, sort_keys=True),
            flush=True,
        )


def _ensure_probe_hf_bucket_runtime(argv: Sequence[str]) -> None:
    """Re-exec this probe launcher, not the deep-decay launcher, with HF Hub 1.5."""

    if deep_decay._hf_bucket_api_available():
        return

    runtime = deep_decay._impl.WORK_ROOT / ".runtime" / f"huggingface-hub-{HF_HUB_VERSION}"
    marker = runtime / ".complete"
    if not marker.is_file():
        staging = runtime.with_name(runtime.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--target",
                str(staging),
                f"huggingface_hub=={HF_HUB_VERSION}",
            ]
        )
        (staging / ".complete").write_text(HF_HUB_VERSION + "\n", encoding="utf-8")
        shutil.rmtree(runtime, ignore_errors=True)
        os.replace(staging, runtime)

    env = dict(os.environ)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(runtime) + (os.pathsep + previous if previous else "")
    print(
        f"[kaggle-probe-a] Kaggle host lacks HF Storage Buckets API; "
        f"restarting this probe launcher with private huggingface_hub=={HF_HUB_VERSION}",
        flush=True,
    )
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *list(argv)],
        env,
    )


def _noop_hf_runtime_restart(argv: Sequence[str]) -> None:
    """Disable the imported deep-decay shim restart after this launcher handled it."""

    return None


def _fixed_source_step(impl: Any) -> int:
    return int(impl._impl._checkpoint_step(PROBE_SOURCE_CHECKPOINT_ID))


def _fixed_source_repo_id(runtime_base: Any, impl: Any) -> str:
    del runtime_base
    explicit = os.environ.get("SMALL_LLM_PROBE_A_SOURCE_REPO_ID", "").strip()
    if explicit:
        return explicit
    return f"{PROBE_BASE_HF_REPO_ID}-best-{impl._impl.RUN_ID}"


def _best_checkpoint_prefix(impl: Any) -> str:
    return f"models/{impl._impl.RUN_ID}/{PROBE_SOURCE_CHECKPOINT_ID}"


def _load_best_source_marker(*, repo_id: str, token: str | None, impl: Any) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    marker_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="best_model.json",
            token=token,
            force_download=True,
        )
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping):
        raise RuntimeError("Probe A best_model.json is not a JSON object")
    if marker.get("role") != "small-llm-dedicated-best-model":
        raise RuntimeError(f"{repo_id!r} is not a Small-LLM dedicated best-model repo")
    if marker.get("run_id") != impl._impl.RUN_ID:
        raise RuntimeError(
            f"Probe A source repo belongs to {marker.get('run_id')!r}, "
            f"not {impl._impl.RUN_ID!r}"
        )
    if marker.get("checkpoint_id") != PROBE_SOURCE_CHECKPOINT_ID:
        raise RuntimeError(
            f"Probe A requires {PROBE_SOURCE_CHECKPOINT_ID}, but best_model.json "
            f"points to {marker.get('checkpoint_id')!r}"
        )
    if marker.get("artifact_path") != _best_checkpoint_prefix(impl):
        raise RuntimeError("Probe A best_model.json artifact_path disagrees with fixed source")
    return dict(marker)


def _ensure_best_source_local_manifest(root: Path) -> bool:
    """Create the local checkpoint manifest missing from some best-model artifacts.

    The dedicated best-model repository is marker-verified and the downloaded
    trainer state is validated again before training. Some best snapshots can be
    downloaded without ``local_manifest.json`` at the checkpoint root, so Probe A
    reconstructs the manifest required by the standard checkpoint verifier.

    ``root`` must be a materialized local copy, not the Hugging Face cache path.
    """

    manifest_path = root / "local_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError(
            "Probe A source local_manifest.json is still a symlink after materialization"
        )
    if manifest_path.is_file():
        return False

    names: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                f"Probe A source checkpoint still contains a symlink after materialization: "
                f"{path.relative_to(root)}"
            )
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name in _POST_SAVE_METADATA:
            continue
        names.append(name)

    required = {"trainer_state.pkl", "checkpoint.json"}
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(
            "Probe A cannot reconstruct local_manifest.json; missing checkpoint "
            f"files: {missing}"
        )

    from dataset.src.remote import sha256_path

    payload = {
        "version": 1,
        "files": [
            {
                "name": name,
                "sha256": sha256_path(root / name),
                "byte_size": (root / name).stat().st_size,
            }
            for name in names
        ],
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    print(
        json.dumps(
            {"probe_a_rebuilt_local_manifest": {"root": str(root), "files": names}},
            sort_keys=True,
        ),
        flush=True,
    )
    return True


def _materialize_best_source_checkpoint(source: Path, staging: Path) -> None:
    """Copy an HF snapshot checkpoint tree into real local files.

    ``snapshot_download`` commonly exposes repo files through symlinks into the
    Hugging Face cache. Checkpoint verification intentionally rejects symlinks, so
    Probe A must dereference-copy the snapshot before hashing or loading it.
    """

    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging, symlinks=False)
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                f"Probe A materialized checkpoint still contains a symlink: "
                f"{path.relative_to(staging)}"
            )


def _restore_fixed_best_checkpoint(runtime_base: Any, impl: Any) -> dict[str, object]:
    """Restore exactly step-00071750 from the dedicated best-model repo.

    This deliberately ignores rolling latest. Probe A must compare LR resets from
    one fixed checkpoint even if the control run has a newer Bucket latest.
    """

    from dataset.src.joint_checkpoint import verify_local_manifest
    from huggingface_hub import snapshot_download

    checkpoint_id = PROBE_SOURCE_CHECKPOINT_ID
    step = _fixed_source_step(impl)
    repo_id = _fixed_source_repo_id(runtime_base, impl)
    prefix = _best_checkpoint_prefix(impl)
    target = impl._impl.CHECKPOINT_DIR / checkpoint_id

    if target.exists():
        verify_local_manifest(target)
        return {
            "checkpoint_id": checkpoint_id,
            "step": step,
            "source": "local_fixed_best",
            "repo_id": repo_id,
            "path_in_repo": prefix,
            "local_manifest_rebuilt": False,
            "materialized_from_hf_cache": False,
        }

    token = runtime_base._hf_token()
    marker = _load_best_source_marker(repo_id=repo_id, token=token, impl=impl)
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            allow_patterns=[f"{prefix}/*", "best_model.json"],
            force_download=True,
        )
    )
    source = snapshot_root / prefix
    if not source.is_dir():
        raise RuntimeError(f"Probe A source checkpoint {repo_id}/{prefix} was not downloaded")

    impl._impl.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    staging = impl._impl.CHECKPOINT_DIR / f".{checkpoint_id}.probe-a-fixed-best"
    _materialize_best_source_checkpoint(source, staging)
    rebuilt_manifest = _ensure_best_source_local_manifest(staging)
    verify_local_manifest(staging)
    os.replace(staging, target)
    verify_local_manifest(target)

    return {
        "checkpoint_id": checkpoint_id,
        "step": step,
        "source": "hf_dedicated_best_model_repo",
        "repo_id": repo_id,
        "path_in_repo": prefix,
        "validation_loss": marker.get("validation_loss"),
        "local_manifest_rebuilt": rebuilt_manifest,
        "materialized_from_hf_cache": True,
    }


def _prepare_fixed_source_checkpoint(runtime_base: Any, impl: Any) -> tuple[str, int, Path]:
    restored = _restore_fixed_best_checkpoint(runtime_base, impl)
    checkpoint_id = str(restored["checkpoint_id"])
    step = int(restored["step"])
    expected_step = _fixed_source_step(impl)
    if checkpoint_id != PROBE_SOURCE_CHECKPOINT_ID or step != expected_step:
        raise RuntimeError("Probe A fixed source identity changed during restore")

    dataset = impl._impl._stage_dataset(runtime_base, start_block_id=min(step, impl._impl.FINAL_STEP - 1))
    migrated = deep_decay._install_execution_migration(checkpoint_id=checkpoint_id, dataset=dataset)
    verified_step = impl._impl._verify_deep_decay_checkpoint(runtime_base, checkpoint_id)
    if verified_step != step:
        raise RuntimeError("verified fixed Probe A source step disagrees with requested step")

    print(
        json.dumps(
            {
                "probe_a_source": {
                    "checkpoint_id": checkpoint_id,
                    "step": step,
                    "source_kind": PROBE_SOURCE_KIND,
                    "base_hf_repo_id": PROBE_BASE_HF_REPO_ID,
                    "repo_id": restored.get("repo_id"),
                    "path_in_repo": restored.get("path_in_repo"),
                    "restored_from": restored.get("source"),
                    "validation_loss": restored.get("validation_loss"),
                    "local_manifest_rebuilt": restored.get("local_manifest_rebuilt"),
                    "materialized_from_hf_cache": restored.get("materialized_from_hf_cache"),
                    "migrated_to_kaggle_execution": bool(migrated),
                    "dataset": str(dataset),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return checkpoint_id, step, dataset


def _patch_impl_dry_run(impl: Any) -> None:
    original_dry_run_payload = impl._dry_run_payload

    def fixed_dry_run_payload(args: Any) -> dict[str, object]:
        payload = dict(original_dry_run_payload(args))
        source_step = _fixed_source_step(impl)
        payload.update(
            {
                "source": PROBE_SOURCE_KIND,
                "source_checkpoint_id": PROBE_SOURCE_CHECKPOINT_ID,
                "source_step": source_step,
                "base_hf_repo_id": PROBE_BASE_HF_REPO_ID,
                "source_repo": "$SMALL_LLM_PROBE_A_SOURCE_REPO_ID or "
                + f"{PROBE_BASE_HF_REPO_ID}-best-"
                + impl._impl.RUN_ID,
                "source_path_in_repo": _best_checkpoint_prefix(impl),
                "hf_reads": [
                    "fixed dedicated best-model checkpoint restore",
                    "rolling 10B dataset shards",
                ],
            }
        )
        for item in payload.get("branches", []):
            if isinstance(item, dict) and isinstance(item.get("branch"), str):
                item["wandb_run_id_template"] = (
                    f"100m-10b-probe-a-{item['branch']}-from-step{source_step}"
                )
        return payload

    impl._dry_run_payload = fixed_dry_run_payload


def _patch_impl_probe_config_canonicalization(impl: Any) -> None:
    original_probe_config = impl._probe_config

    def canonical_probe_config(
        original_config: Mapping[str, object],
        *,
        branch: Any,
        eval_every_steps: int,
    ) -> dict[str, object]:
        from trainer.config import TrainerConfig

        raw = dict(
            original_probe_config(
                original_config,
                branch=branch,
                eval_every_steps=eval_every_steps,
            )
        )
        canonical = TrainerConfig(**raw).as_dict()
        dropped = sorted(set(raw) - set(canonical))
        if dropped:
            print(
                json.dumps(
                    {
                        "probe_a_canonicalized_branch_config": {
                            "branch": getattr(branch, "slug", "unknown"),
                            "dropped_constant_schedule_keys": dropped,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return canonical

    impl._probe_config = canonical_probe_config


def _patch_impl_wandb_resume_allow(impl: Any) -> None:
    original_prefix = impl._wandb_env_prefix
    original_build = impl._build_branch_trainer_command
    original_assert = impl._assert_branch_wandb_identity

    def wandb_env_prefix_allow(branch: Any, source_step: int, branch_run_dir: Path) -> list[str]:
        command = list(original_prefix(branch, source_step, branch_run_dir))
        return ["WANDB_RESUME=allow" if item == "WANDB_RESUME=must" else item for item in command]

    def assert_branch_wandb_identity_allow(command: Sequence[str], *, branch: Any, source_step: int) -> None:
        expected = impl._branch_run_id(branch, source_step)
        values = impl._all_option_values(command, "--wandb-run-id")
        if values != [expected]:
            raise RuntimeError(
                f"Probe A branch {branch.slug} must have exactly one W&B run ID "
                f"{expected!r}; got {values!r}"
            )
        resumes = impl._all_option_values(command, "--wandb-resume")
        if resumes != ["allow"]:
            raise RuntimeError(
                f"Probe A branch {branch.slug} must use --wandb-resume allow; got {resumes!r}"
            )

    def build_branch_trainer_command_allow(*args: Any, **kwargs: Any) -> list[str]:
        previous_assert = impl._assert_branch_wandb_identity
        impl._assert_branch_wandb_identity = original_assert
        try:
            command = list(original_build(*args, **kwargs))
        finally:
            impl._assert_branch_wandb_identity = previous_assert
        impl._replace_option(command, "--wandb-resume", "allow")
        branch = kwargs["branch"]
        source_step = int(kwargs["source_step"])
        assert_branch_wandb_identity_allow(command, branch=branch, source_step=source_step)
        return command

    impl._wandb_env_prefix = wandb_env_prefix_allow
    impl._assert_branch_wandb_identity = assert_branch_wandb_identity_allow
    impl._build_branch_trainer_command = build_branch_trainer_command_allow


def _patch_impl_for_fixed_source(impl: Any) -> None:
    impl.PROBE_SOURCE_CHECKPOINT_ID = PROBE_SOURCE_CHECKPOINT_ID
    impl.PROBE_SOURCE_STEP = _fixed_source_step(impl)
    impl.PROBE_SOURCE_KIND = PROBE_SOURCE_KIND
    impl.PROBE_BASE_HF_REPO_ID = PROBE_BASE_HF_REPO_ID
    impl._prepare_source_checkpoint = lambda runtime_base: _prepare_fixed_source_checkpoint(runtime_base, impl)
    _patch_impl_dry_run(impl)
    _patch_impl_probe_config_canonicalization(impl)
    _patch_impl_wandb_resume_allow(impl)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _force_probe_hf_identity()
    if "--dry-run" not in args:
        _ensure_probe_hf_bucket_runtime(args)

    # The implementation imports the deep-decay shim for checkpoint migration,
    # but it must never call that shim's self-reexec from Probe A. If the host
    # still lacks Storage Bucket support after the probe restart, downstream HF
    # calls should fail in-place instead of jumping to the deep-decay trainer.
    deep_decay._ensure_host_hf_bucket_runtime = _noop_hf_runtime_restart

    import probe_a_lr_reset_10b_impl as impl

    _patch_impl_for_fixed_source(impl)
    return int(impl.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
