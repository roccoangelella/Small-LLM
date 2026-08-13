#!/usr/bin/env python3
"""Privately publish and round-trip verify one immutable SFT bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Mapping, Sequence

from post_training.sft.bundle import verify_bundle


class PublishFailure(RuntimeError):
    pass


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise PublishFailure(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise PublishFailure(f"{label} is not a JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise PublishFailure(f"unsafe SFT bundle root: {root}")
    digest = hashlib.sha256()
    total = 0
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PublishFailure(f"SFT bundle contains a symlink: {path}")
        if path.is_file():
            files.append(path)
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = _sha256(path)
        total += size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


def _copy_or_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _stage(bundle: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(bundle, destination, copy_function=_copy_or_link)
    verify_bundle(destination)
    return tree_identity(destination)


def _anonymous_access(handle: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="small-llm-sft-anonymous-") as temporary:
        root = Path(temporary)
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"}
        }
        env.update(
            HOME=str(root / "home"),
            KAGGLE_CONFIG_DIR=str(root / "config"),
            KAGGLEHUB_CACHE=str(root / "cache"),
            XDG_CACHE_HOME=str(root / "xdg"),
        )
        code = (
            "import kagglehub,sys; kagglehub.dataset_download(" 
            "sys.argv[1], path='bundle-manifest.json', output_dir=sys.argv[2], force_download=True)"
        )
        return subprocess.run(
            [sys.executable, "-c", code, handle, str(root / "download")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0


def _downloaded_root(returned: str, requested: Path) -> Path:
    candidates: set[Path] = set()
    for path in (Path(returned), requested):
        if path.is_dir() and (path / "bundle-manifest.json").is_file():
            candidates.add(path.resolve())
    if requested.exists():
        candidates.update(
            path.parent.resolve()
            for path in requested.rglob("bundle-manifest.json")
            if path.is_file()
        )
    if len(candidates) != 1:
        raise PublishFailure(
            f"cannot identify exactly one Kaggle SFT round-trip root: {sorted(candidates)}"
        )
    return candidates.pop()


def _remove_kagglehub_transport_artifacts(root: Path) -> tuple[str, ...]:
    """Remove downloader-owned root artifacts before byte-for-byte dataset comparison.

    KaggleHub's direct ``output_dir`` resolver may leave the downloaded version archive
    (for example ``1.archive``) and a zero-byte completion marker tree beside the
    extracted dataset files. Those are transport implementation details, not dataset
    files, and must not participate in the immutable bundle tree identity.
    """

    removed: list[str] = []
    for path in sorted(root.glob("*.archive")):
        if path.is_symlink() or not path.is_file():
            continue
        path.unlink()
        removed.append(path.name)

    marker_root = root / ".complete"
    if marker_root.exists():
        if marker_root.is_symlink() or not marker_root.is_dir():
            raise PublishFailure(f"unsafe KaggleHub completion-marker root: {marker_root}")
        marker_files: list[Path] = []
        for path in sorted(marker_root.rglob("*")):
            if path.is_symlink():
                raise PublishFailure(f"unsafe KaggleHub completion-marker symlink: {path}")
            if path.is_file():
                if path.suffix != ".complete" or path.stat().st_size != 0:
                    raise PublishFailure(f"unexpected KaggleHub completion-marker content: {path}")
                marker_files.append(path)
            elif not path.is_dir():
                raise PublishFailure(f"unexpected KaggleHub completion-marker entry: {path}")
        removed.extend(path.relative_to(root).as_posix() for path in marker_files)
        shutil.rmtree(marker_root)
    return tuple(removed)


def _roundtrip(
    handle: str,
    *,
    destination: Path,
    expected: Mapping[str, object],
    timeout_seconds: int,
) -> dict[str, object]:
    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover
        raise PublishFailure("kagglehub is required for SFT publication") from error
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        try:
            returned = kagglehub.dataset_download(
                handle,
                output_dir=str(destination),
                force_download=True,
            )
            root = _downloaded_root(str(returned), destination)
            removed = _remove_kagglehub_transport_artifacts(root)
            if removed:
                print(
                    f"[sft-publish] removed KaggleHub transport artifacts: {', '.join(removed)}",
                    flush=True,
                )
            identity = tree_identity(root)
            if identity.get("tree_sha256") != expected.get("tree_sha256"):
                raise PublishFailure(
                    "Kaggle SFT round-trip tree differs from staged bytes: "
                    f"expected={expected.get('tree_sha256')} actual={identity.get('tree_sha256')} "
                    f"expected_files={expected.get('file_count')} actual_files={identity.get('file_count')} "
                    f"expected_bytes={expected.get('total_bytes')} actual_bytes={identity.get('total_bytes')}"
                )
            verification = verify_bundle(root)
            if _anonymous_access(handle):
                raise PublishFailure("uploaded SFT Kaggle dataset is publicly readable")
            return {
                "status": "passed",
                "root": str(root),
                "identity": identity,
                "verification": verification,
                "anonymous_access": "denied",
            }
        except Exception as error:  # noqa: BLE001
            last_error = error
            print(
                f"[sft-publish] round-trip attempt {attempt} failed: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(15, remaining))
    raise PublishFailure(f"Kaggle SFT dataset did not become verifiably downloadable: {last_error}")


def _state_matches(
    state: Mapping[str, object],
    *,
    handle: str,
    identity: Mapping[str, object],
    bundle_manifest_sha256: str,
) -> bool:
    return (
        state.get("handle") == handle
        and state.get("tree_sha256") == identity.get("tree_sha256")
        and state.get("bundle_manifest_sha256") == bundle_manifest_sha256
    )


def _complete_roundtrip_state(
    state: Mapping[str, object],
    *,
    state_path: Path,
    summary_path: Path,
    summary: Mapping[str, object],
    handle: str,
    roundtrip: Path,
    expected: Mapping[str, object],
    timeout_seconds: int,
) -> int:
    remote = _roundtrip(
        handle,
        destination=roundtrip,
        expected=expected,
        timeout_seconds=timeout_seconds,
    )
    verified = {**dict(state), "status": "verified", "remote": remote}
    _write_json(state_path, verified)
    completed = {**dict(summary), "status": "completed", "remote": remote}
    _write_json(summary_path, completed)
    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--handle", required=True)
    parser.add_argument("--ops-dir", type=Path, required=True)
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument("--remote-ready-timeout-seconds", type=int, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.remote_ready_timeout_seconds <= 0:
        raise SystemExit("--remote-ready-timeout-seconds must be positive")
    if "/" not in args.handle or args.handle.count("/") != 1:
        raise SystemExit("--handle must use owner/dataset form")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is required for private SFT bundle publication")

    bundle = args.dataset_dir.resolve()
    verification = verify_bundle(bundle)
    bundle_manifest = _read_json(bundle / "bundle-manifest.json", label="SFT bundle manifest")
    bundle_hash = bundle_manifest.get("manifest_sha256")
    if not isinstance(bundle_hash, str) or len(bundle_hash) != 64:
        raise PublishFailure("SFT bundle has no valid manifest identity")

    ops = args.ops_dir.resolve()
    stage = ops / "kaggle-dataset"
    roundtrip = ops / "kaggle-roundtrip"
    state_path = ops / "publish-state.json"
    summary_path = ops / "publish-summary.json"
    ops.mkdir(parents=True, exist_ok=True)

    staged_identity = _stage(bundle, stage)
    summary: dict[str, object] = {
        "schema": "small-llm-sft-kaggle-publication-v1",
        "handle": args.handle,
        "bundle": verification,
        "bundle_manifest_sha256": bundle_hash,
        "staged_identity": staged_identity,
    }
    previous = _read_json(state_path, label="SFT publication state") if state_path.is_file() else {}
    matching_previous = _state_matches(
        previous,
        handle=args.handle,
        identity=staged_identity,
        bundle_manifest_sha256=bundle_hash,
    )
    if not args.force_upload and matching_previous:
        status = previous.get("status")
        if status == "verified":
            if _anonymous_access(args.handle):
                raise PublishFailure("previously published SFT Kaggle dataset became public")
            summary.update(status="already_published", remote=previous.get("remote"))
            _write_json(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if status in {"upload_attempting", "upload_submitted"}:
            try:
                return _complete_roundtrip_state(
                    previous,
                    state_path=state_path,
                    summary_path=summary_path,
                    summary=summary,
                    handle=args.handle,
                    roundtrip=roundtrip,
                    expected=staged_identity,
                    timeout_seconds=args.remote_ready_timeout_seconds,
                )
            except PublishFailure:
                if status == "upload_submitted":
                    raise
                # upload_attempting may have been recorded before the request was
                # accepted remotely. Fall through to one idempotent upload attempt.

    if _anonymous_access(args.handle):
        raise PublishFailure(f"refusing publicly readable Kaggle handle: {args.handle}")
    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover
        raise PublishFailure("kagglehub is required for SFT publication") from error

    attempt = {
        "schema": "small-llm-sft-kaggle-publication-state-v1",
        "status": "upload_attempting",
        "handle": args.handle,
        "bundle_manifest_sha256": bundle_hash,
        **staged_identity,
    }
    _write_json(state_path, attempt)
    kagglehub.dataset_upload(
        args.handle,
        str(stage),
        version_notes=(
            f"Small-LLM SFT bundle {bundle_hash}; tree {staged_identity['tree_sha256']}"
        ),
    )
    submitted = {**attempt, "status": "upload_submitted"}
    _write_json(state_path, submitted)
    return _complete_roundtrip_state(
        submitted,
        state_path=state_path,
        summary_path=summary_path,
        summary=summary,
        handle=args.handle,
        roundtrip=roundtrip,
        expected=staged_identity,
        timeout_seconds=args.remote_ready_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PublishFailure",
    "_remove_kagglehub_transport_artifacts",
    "_state_matches",
    "build_parser",
    "main",
    "tree_identity",
]
