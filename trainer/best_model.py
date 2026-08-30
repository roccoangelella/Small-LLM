"""Dedicated Hugging Face best-model publication with replace-only history.

This transport is intentionally separate from exact-resume checkpoints.  A caller
selects a checkpoint by validation loss, and this module publishes that verified
joint checkpoint to one dedicated model repository.  Updating an existing best
repository is allowed only by deleting and recreating the repository first, so
training checkpoints never accumulate Git/LFS history there.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from dataset.src.joint_checkpoint import verify_local_manifest

_BEST_MARKER = "best_model.json"
_BEST_ROLE = "small-llm-dedicated-best-model"
_CHECKPOINT_ID = re.compile(r"^step-(\d{8})$")


def _validate_repo_id(repo_id: str) -> str:
    if not isinstance(repo_id, str):
        raise RuntimeError("best-model repository ID must be a string")
    value = repo_id.strip()
    owner, separator, name = value.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise RuntimeError("best-model repository ID must be in owner/name form")
    return value


def _validate_identity(run_id: str, checkpoint_id: str) -> tuple[str, str]:
    if not isinstance(run_id, str) or not run_id.strip() or "/" in run_id or "\\" in run_id:
        raise RuntimeError(f"invalid best-model run_id: {run_id!r}")
    run_id = run_id.strip()
    if not isinstance(checkpoint_id, str) or _CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
        raise RuntimeError(f"invalid best-model checkpoint_id: {checkpoint_id!r}")
    return run_id, checkpoint_id


def _finite_metric(metric: object, *, label: str) -> float:
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    value = float(metric)
    if not math.isfinite(value):
        raise RuntimeError(f"{label} is non-finite")
    return value


def _repo_exists(api: Any, *, repo_id: str, token: str | None) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
    except Exception as error:  # Hugging Face has changed the concrete 404 class over time.
        status = getattr(getattr(error, "response", None), "status_code", None)
        if error.__class__.__name__ in {"RepositoryNotFoundError", "RepoNotFoundError"} or status == 404:
            return False
        raise
    return True


def _download_json(*, repo_id: str, path: str, token: str | None) -> Mapping[str, object]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("best-model publication requires huggingface_hub") from error
    try:
        local = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=path,
                token=token,
                force_download=True,
            )
        )
    except Exception as error:
        if error.__class__.__name__ in {
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
            "LocalEntryNotFoundError",
        }:
            raise RuntimeError(
                f"existing best-model repository {repo_id!r} is missing required marker {path!r}"
            ) from error
        raise
    payload = json.loads(local.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"best-model marker {path!r} is not a JSON object")
    return dict(payload)


def _validated_marker(
    payload: Mapping[str, object],
    *,
    repo_id: str,
    run_id: str,
) -> dict[str, object]:
    if payload.get("version") != 1 or payload.get("role") != _BEST_ROLE:
        raise RuntimeError(f"repository {repo_id!r} is not a Small-LLM dedicated best-model repository")
    if payload.get("run_id") != run_id:
        raise RuntimeError(
            f"repository {repo_id!r} belongs to run {payload.get('run_id')!r}, not {run_id!r}"
        )
    checkpoint_id = payload.get("checkpoint_id")
    _validate_identity(run_id, checkpoint_id)  # type: ignore[arg-type]
    metric = _finite_metric(payload.get("metric"), label="best-model marker metric")
    validation_loss = _finite_metric(
        payload.get("validation_loss"),
        label="best-model marker validation_loss",
    )
    if validation_loss < 0 or metric != -validation_loss:
        raise RuntimeError("best-model marker metric disagrees with validation loss")
    return dict(payload)


def get_dedicated_best_metric(
    *,
    repo_id: str,
    run_id: str,
    token: str | None,
) -> float | None:
    """Return the persisted higher-is-better metric, or ``None`` if repo is absent."""

    repo_id = _validate_repo_id(repo_id)
    run_id, _ = _validate_identity(run_id, "step-00000000")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("best-model publication requires huggingface_hub") from error
    api = HfApi(token=token)
    if not _repo_exists(api, repo_id=repo_id, token=token):
        return None
    marker = _validated_marker(
        _download_json(repo_id=repo_id, path=_BEST_MARKER, token=token),
        repo_id=repo_id,
        run_id=run_id,
    )
    return _finite_metric(marker["metric"], label="best-model marker metric")


def checkpoint_validation_loss(checkpoint_dir: Path) -> float:
    payload_path = checkpoint_dir / "checkpoint.json"
    if payload_path.is_symlink() or not payload_path.is_file():
        raise RuntimeError("best-model checkpoint.json is missing or not a regular file")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("best-model checkpoint.json is not valid JSON") from error
    metrics = payload.get("validation_metrics") if isinstance(payload, Mapping) else None
    loss = metrics.get("loss") if isinstance(metrics, Mapping) else None
    value = _finite_metric(loss, label="best-model checkpoint validation loss")
    if value < 0:
        raise RuntimeError("best-model checkpoint validation loss cannot be negative")
    return value


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise RuntimeError(f"best-model checkpoint is not a real directory: {checkpoint_dir}")
    files: list[Path] = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"best-model checkpoint contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"best-model checkpoint contains a non-regular file: {path}")
        files.append(path)
    return files


def publish_dedicated_best_model(
    *,
    repo_id: str,
    run_id: str,
    checkpoint_dir: Path,
    checkpoint_id: str,
    metric: float,
    validation_loss: float,
    token: str | None,
    recreate: bool,
) -> dict[str, object]:
    """Publish one verified best checkpoint, recreating an existing repo first.

    Existing repositories are deleted only after their ownership marker proves that
    they are the dedicated best repository for ``run_id``.  This is the safety gate
    that makes the requested delete/recreate update policy non-ambiguous.
    """

    repo_id = _validate_repo_id(repo_id)
    run_id, checkpoint_id = _validate_identity(run_id, checkpoint_id)
    metric_value = _finite_metric(metric, label="best-model metric")
    loss_value = _finite_metric(validation_loss, label="best-model validation_loss")
    if loss_value < 0 or metric_value != -loss_value:
        raise RuntimeError("best-model metric must be exactly the negated validation loss")
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.name != checkpoint_id:
        raise RuntimeError("best-model checkpoint directory name disagrees with checkpoint_id")
    verify_local_manifest(checkpoint_dir)
    checkpoint_loss = checkpoint_validation_loss(checkpoint_dir)
    if checkpoint_loss != loss_value:
        raise RuntimeError(
            "best-model checkpoint validation loss disagrees with the selected validation metric"
        )
    files = _checkpoint_files(checkpoint_dir)

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as error:
        raise RuntimeError("best-model publication requires huggingface_hub") from error

    api = HfApi(token=token)
    existed = _repo_exists(api, repo_id=repo_id, token=token)
    previous: dict[str, object] | None = None
    if existed:
        previous = _validated_marker(
            _download_json(repo_id=repo_id, path=_BEST_MARKER, token=token),
            repo_id=repo_id,
            run_id=run_id,
        )
        if not recreate:
            raise RuntimeError(
                "best-model repository already exists; delete/recreate is required for every update"
            )
        api.delete_repo(repo_id=repo_id, repo_type="model", token=token, missing_ok=False)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        token=token,
        exist_ok=False,
    )

    prefix = f"models/{run_id}/{checkpoint_id}"
    artifact_path = f"models/{run_id}/artifact.json"
    marker = {
        "version": 1,
        "role": _BEST_ROLE,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "metric": metric_value,
        "validation_loss": loss_value,
        "selection": "strict_validation_loss_improvement",
        "replacement": "delete_recreate_repository",
        "artifact_path": prefix,
    }
    artifact = {
        "version": 1,
        "artifact_type": "small-llm-best-joint-checkpoint",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "huggingface_path": prefix,
        "validation_loss": loss_value,
        "metric": metric_value,
        "verification": "dataset.src.joint_checkpoint.verify_local_manifest passed before upload",
    }
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{path.relative_to(checkpoint_dir).as_posix()}",
            path_or_fileobj=str(path),
        )
        for path in files
    ]
    operations.extend(
        [
            CommitOperationAdd(
                path_in_repo=artifact_path,
                path_or_fileobj=BytesIO((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode()),
            ),
            CommitOperationAdd(
                path_in_repo=_BEST_MARKER,
                path_or_fileobj=BytesIO((json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()),
            ),
        ]
    )
    try:
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Publish best {run_id} {checkpoint_id}",
            token=token,
        )
        observed = _validated_marker(
            _download_json(repo_id=repo_id, path=_BEST_MARKER, token=token),
            repo_id=repo_id,
            run_id=run_id,
        )
        if observed != marker:
            raise RuntimeError("best-model marker read-back mismatch after publication")
    except BaseException:
        # A failed replacement must never leave an apparently valid but partial
        # best repository. Deleting it is safe because this invocation either
        # created it from scratch or recreated a marker-verified dedicated repo.
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model", token=token, missing_ok=True)
        except Exception:
            pass
        raise

    return {
        "status": "replaced" if existed else "published",
        "repo_id": repo_id,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "validation_loss": loss_value,
        "metric": metric_value,
        "path_in_repo": prefix,
        "metadata_path": artifact_path,
        "marker_path": _BEST_MARKER,
        "commit": getattr(commit, "oid", None),
        "previous_checkpoint_id": None if previous is None else previous.get("checkpoint_id"),
        "previous_metric": None if previous is None else previous.get("metric"),
    }


__all__ = [
    "checkpoint_validation_loss",
    "get_dedicated_best_metric",
    "publish_dedicated_best_model",
]
