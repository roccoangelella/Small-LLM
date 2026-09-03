#!/usr/bin/env python3
"""Build an HF-durable dataset, verify it, then privately publish it to Kaggle.

The filename is historical because ``kaggle/runtime.py`` reuses this engine for
all finite 20M data profiles. New remote durability is Hugging Face Storage
Bucket only. The staged Kaggle tree still includes ``drive_manifest.json`` as a
legacy compatibility filename consumed by existing trainer/checkpoint code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
PROFILE = "20m-100m-data-scaling-v1"
RUN_ID = "20m-100m-dataset-001"
SLUG = "small-llm-20m-100m-dataset-001"
DEFAULT_WEIGHTS = "/data/climbmix-mixture-calibration/climbmix_code_free_weights.json"
DEFAULT_DATASET = "/data/small-llm/20m-100m-dataset-001"
DEFAULT_OPS = "/data/small-llm/20m-100m-ops"
FILES = ("manifest.json", "drive_manifest.json", "qualification_plan.json")
DIRS = ("train", "validation")
HANDLE_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")


class SuiteFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Config:
    weights: Path
    dataset: Path
    ops: Path
    handle: str
    force_upload: bool
    timeout: int

    @property
    def logs(self) -> Path:
        return self.ops / "logs"

    @property
    def publish(self) -> Path:
        return self.ops / "kaggle-dataset"

    @property
    def roundtrip(self) -> Path:
        return self.ops / "kaggle-roundtrip"

    @property
    def state(self) -> Path:
        return self.ops / "kaggle-publish-state.json"

    @property
    def summary(self) -> Path:
        return self.ops / "build-and-push-summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise SuiteFailure(f"Cannot read {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise SuiteFailure(f"{label} is not a JSON object: {path}")
    return dict(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_handle(explicit: str | None, env: Mapping[str, str]) -> str:
    handle = explicit or env.get("SMALL_LLM_KAGGLE_DATASET_HANDLE", "")
    if not handle and env.get("KAGGLE_USERNAME"):
        handle = f"{env['KAGGLE_USERNAME']}/{SLUG}"
    if not HANDLE_RE.fullmatch(handle):
        raise SuiteFailure(
            "Set KAGGLE_USERNAME or SMALL_LLM_KAGGLE_DATASET_HANDLE=owner/dataset in .env"
        )
    return handle


def arguments(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=Path, default=env.get("SMALL_LLM_100M_WEIGHTS_FILE", DEFAULT_WEIGHTS))
    parser.add_argument("--dataset-dir", type=Path, default=env.get("SMALL_LLM_100M_DATASET_DIR", DEFAULT_DATASET))
    parser.add_argument("--ops-dir", type=Path, default=env.get("SMALL_LLM_100M_OPS_DIR", DEFAULT_OPS))
    parser.add_argument("--kaggle-dataset-handle")
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument(
        "--remote-ready-timeout-seconds",
        type=int,
        default=int(env.get("SMALL_LLM_KAGGLE_READY_TIMEOUT_SECONDS", "900")),
    )
    args = parser.parse_args(argv)
    if args.remote_ready_timeout_seconds <= 0:
        raise SuiteFailure("Remote-ready timeout must be positive")
    return Config(
        weights=args.weights_file.expanduser().resolve(),
        dataset=args.dataset_dir.expanduser().resolve(),
        ops=args.ops_dir.expanduser().resolve(),
        handle=resolve_handle(args.kaggle_dataset_handle, env),
        force_upload=args.force_upload,
        timeout=args.remote_ready_timeout_seconds,
    )


def check_environment(config: Config) -> str:
    for name in ("KAGGLE_API_TOKEN", "HF_TOKEN"):
        if not os.environ.get(name):
            raise SuiteFailure(f"Missing required .env value: {name}")
    if not (
        os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    ):
        raise SuiteFailure(
            "Set SMALL_LLM_HF_DATASET_BUCKET_ID or SMALL_LLM_HF_REPO_ID for dataset durability"
        )
    if not config.weights.is_file():
        raise SuiteFailure(f"Mixture weights file is missing: {config.weights}")
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=REPO, text=True
        ).strip()
    ).resolve()
    if root != REPO.resolve():
        raise SuiteFailure(f"Repository root mismatch: {root}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True
    ).strip():
        raise SuiteFailure("The repository has tracked modifications")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SuiteFailure("Cannot resolve the source commit")
    config.dataset.parent.mkdir(parents=True, exist_ok=True)
    config.logs.mkdir(parents=True, exist_ok=True)
    return commit


def run(command: Sequence[str], name: str, config: Config) -> None:
    log_path = config.logs / f"{name}.log"
    write_json(
        config.logs / f"{name}.command.json",
        {"command": list(command), "started_utc": now()},
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    (config.logs / f"{name}.exit-code").write_text(f"{code}\n", encoding="utf-8")
    if code:
        raise SuiteFailure(f"{name} failed with exit code {code}; see {log_path}")


def production_identity() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "target_source_tokens": 100_000_000,
        "minimum_source_tokens": 90_000_000,
        "maximum_source_tokens": 110_000_000,
        "checkpoint_source_tokens": 20_000_000,
        "target_reached": True,
        "remote_required": True,
    }


def dataset_complete(root: Path) -> bool:
    if not (root / "manifest.json").is_file():
        return False
    try:
        production = read_json(root / "manifest.json", "dataset manifest").get("production")
    except SuiteFailure:
        return False
    return isinstance(production, Mapping) and all(
        production.get(key) == value for key, value in production_identity().items()
    )


def producer_command(config: Config, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dataset.qualification",
        "build",
        "--profile",
        "20m-100m",
        "--weights-file",
        str(config.weights),
        "--output-dir",
        str(config.dataset),
    ]
    return command + (["--resume"] if resume else [])


def build(config: Config) -> str:
    if dataset_complete(config.dataset):
        print(f"Dataset is already complete: {config.dataset}")
        return "already_complete"
    resume = config.dataset.exists()
    run(
        producer_command(config, resume),
        "dataset-build-resume" if resume else "dataset-build",
        config,
    )
    if not dataset_complete(config.dataset):
        raise SuiteFailure("Producer exited successfully but the fixed dataset is incomplete")
    return "resumed" if resume else "built"


def full_scan(root: Path, prefix: str, config: Config) -> None:
    run(
        [
            sys.executable,
            "-m",
            "dataset.main",
            "verify",
            "--output-dir",
            str(root),
            "--full-scan",
        ],
        f"{prefix}-full-scan",
        config,
    )


def derive_plan(root: Path, prefix: str, config: Config) -> None:
    # Existing Kaggle datasets/checkpoints bind this legacy filename into their
    # identity. It is a provider-neutral durability manifest now.
    run(
        [
            sys.executable,
            "-m",
            "dataset.qualification",
            "report",
            "--profile",
            "20m-100m",
            "--dataset-dir",
            str(root),
            "--drive-manifest",
            str(root / "drive_manifest.json"),
            "--output",
            str(root / "qualification_plan.json"),
        ],
        f"{prefix}-qualification-plan",
        config,
    )


def validate_shape(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise SuiteFailure(f"Unsafe dataset root: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise SuiteFailure("Dataset contains a symlink")
    for name in FILES:
        if not (root / name).is_file():
            raise SuiteFailure(f"Missing training-facing file: {name}")
    for name in DIRS:
        if not (root / name).is_dir():
            raise SuiteFailure(f"Missing training-facing directory: {name}")
    manifest = read_json(root / "manifest.json", "dataset manifest")
    durability = read_json(root / "drive_manifest.json", "legacy durability manifest")
    plan = read_json(root / "qualification_plan.json", "qualification plan")
    expected = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": 2_048,
        "stored_tokens_per_sequence": 2_049,
        "sequences_per_block": 16,
        "target_shard_bytes": 8 * 1024 * 1024,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SuiteFailure("Dataset manifest geometry mismatch")
    production = manifest.get("production")
    if not isinstance(production, Mapping) or any(
        production.get(key) != value for key, value in production_identity().items()
    ):
        raise SuiteFailure("Dataset production identity mismatch")
    shards = durability.get("shards")
    if durability.get("version") != 1 or durability.get("run_id") != RUN_ID:
        raise SuiteFailure("Durability manifest identity mismatch")
    if not isinstance(shards, list) or not shards or any(
        not isinstance(item, Mapping) or item.get("remote_durable") is not True
        for item in shards
    ):
        raise SuiteFailure("Durability manifest contains a non-durable shard")
    identity = plan.get("identity")
    manifest_hash = sha256(root / FILES[0])
    durability_hash = sha256(root / FILES[1])
    if plan.get("qualification_profile") != PROFILE or not isinstance(identity, Mapping):
        raise SuiteFailure("Qualification-plan profile mismatch")
    if (
        identity.get("manifest_sha256") != manifest_hash
        or identity.get("drive_manifest_sha256") != durability_hash
    ):
        raise SuiteFailure("Qualification-plan hashes mismatch")
    return {
        "profile": PROFILE,
        "run_id": RUN_ID,
        "manifest_sha256": manifest_hash,
        "drive_manifest_sha256": durability_hash,
        "qualification_plan_sha256": sha256(root / FILES[2]),
        "accepted_source_tokens": manifest.get("accepted_source_tokens"),
    }


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def stage(config: Config) -> dict[str, object]:
    if config.publish.exists():
        shutil.rmtree(config.publish)
    config.publish.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(config.dataset / name, config.publish / name)
    for name in DIRS:
        shutil.copytree(
            config.dataset / name,
            config.publish / name,
            copy_function=link_or_copy,
        )
    return validate_shape(config.publish)


def tree_identity(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = sha256(path)
        total += size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


def state_matches(
    state: Mapping[str, object],
    config: Config,
    identity: Mapping[str, object],
) -> bool:
    return all(
        (
            state.get("handle") == config.handle,
            state.get("tree_sha256") == identity.get("tree_sha256"),
            state.get("profile") == PROFILE,
            state.get("run_id") == RUN_ID,
        )
    )


def anonymous_access(handle: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="small-llm-kaggle-anonymous-") as temporary:
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
            "sys.argv[1], path='manifest.json', output_dir=sys.argv[2], force_download=True)"
        )
        return subprocess.run(
            [sys.executable, "-c", code, handle, str(root / "download")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0


def upload(config: Config, identity: Mapping[str, object], commit: str) -> None:
    if anonymous_access(config.handle):
        raise SuiteFailure(f"Refusing publicly readable Kaggle handle: {config.handle}")
    try:
        import kagglehub
    except ImportError as error:
        raise SuiteFailure("kagglehub is not installed by the wrapper") from error
    kagglehub.dataset_upload(
        config.handle,
        str(config.publish),
        version_notes=f"{PROFILE}; source {commit}; tree {identity['tree_sha256']}",
    )


def downloaded_root(returned: str, requested: Path) -> Path:
    candidates = set()
    for path in (Path(returned), requested):
        if path.is_dir() and (path / "manifest.json").is_file():
            candidates.add(path.resolve())
    if requested.exists():
        candidates.update(path.parent.resolve() for path in requested.rglob("manifest.json"))
    if len(candidates) != 1:
        raise SuiteFailure(f"Cannot identify one Kaggle round-trip root: {sorted(candidates)}")
    return candidates.pop()


def roundtrip(config: Config, expected: Mapping[str, object]) -> dict[str, object]:
    try:
        import kagglehub
    except ImportError as error:
        raise SuiteFailure("kagglehub is not installed by the wrapper") from error
    deadline, last = time.monotonic() + config.timeout, None
    while time.monotonic() < deadline:
        if config.roundtrip.exists():
            shutil.rmtree(config.roundtrip)
        config.roundtrip.mkdir(parents=True)
        try:
            returned = kagglehub.dataset_download(
                config.handle,
                output_dir=str(config.roundtrip),
                force_download=True,
            )
            root = downloaded_root(returned, config.roundtrip)
            break
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(15)
    else:
        raise SuiteFailure(f"Kaggle dataset did not become downloadable: {last}")
    if tree_identity(root).get("tree_sha256") != expected.get("tree_sha256"):
        raise SuiteFailure("Kaggle round-trip tree differs from the staged tree")
    full_scan(root, "kaggle-roundtrip", config)
    shape = validate_shape(root)
    if anonymous_access(config.handle):
        raise SuiteFailure("Uploaded Kaggle dataset is publicly readable")
    return {
        "status": "passed",
        "root": str(root),
        "shape": shape,
        "anonymous_access": "denied",
    }


def main(argv: Sequence[str] | None = None) -> int:
    config = arguments(argv)
    summary: dict[str, object] = {
        "schema_version": 1,
        "started_utc": now(),
        "status": "initializing",
        "profile": PROFILE,
        "run_id": RUN_ID,
        "handle": config.handle,
        "dataset_dir": str(config.dataset),
        "ops_dir": str(config.ops),
        "dataset_durability": "hf_storage_bucket",
    }
    try:
        commit = check_environment(config)
        summary["source_commit"] = commit
        write_json(config.summary, summary)
        summary["build"] = build(config)
        full_scan(config.dataset, "local", config)
        derive_plan(config.dataset, "local", config)
        summary["local_shape"] = validate_shape(config.dataset)
        summary["publish_shape"] = stage(config)
        identity = tree_identity(config.publish)
        summary["publish_identity"] = identity
        previous = read_json(config.state, "publish state") if config.state.is_file() else {}
        if state_matches(previous, config, identity) and not config.force_upload:
            if previous.get("status") == "verified":
                summary.update(
                    status="already_published",
                    completed_utc=now(),
                    remote=previous.get("remote"),
                )
                write_json(config.summary, summary)
                print(f"Already privately published and verified: {config.handle}")
                return 0
            if previous.get("status") in {"upload_attempting", "upload_submitted"}:
                remote = roundtrip(config, identity)
                write_json(
                    config.state,
                    {
                        **previous,
                        "status": "verified",
                        "verified_utc": now(),
                        "remote": remote,
                    },
                )
                summary.update(status="completed", completed_utc=now(), remote=remote)
                write_json(config.summary, summary)
                print(f"Privately published and verified: {config.handle}")
                return 0
        attempt = {
            "schema_version": 1,
            "status": "upload_attempting",
            "started_utc": now(),
            "profile": PROFILE,
            "run_id": RUN_ID,
            "handle": config.handle,
            "source_commit": commit,
            **identity,
        }
        write_json(config.state, attempt)
        upload(config, identity, commit)
        submitted = {
            **attempt,
            "status": "upload_submitted",
            "submitted_utc": now(),
        }
        write_json(config.state, submitted)
        remote = roundtrip(config, identity)
        write_json(
            config.state,
            {
                **submitted,
                "status": "verified",
                "verified_utc": now(),
                "remote": remote,
            },
        )
        summary.update(status="completed", completed_utc=now(), remote=remote)
        write_json(config.summary, summary)
        print(f"Privately published and verified: {config.handle}")
        print(f"Summary: {config.summary}")
        return 0
    except KeyboardInterrupt:
        summary.update(status="interrupted", finished_utc=now())
        write_json(config.summary, summary)
        return 130
    except Exception as error:  # noqa: BLE001
        summary.update(
            status="failed",
            finished_utc=now(),
            error=f"{type(error).__name__}: {error}",
        )
        write_json(config.summary, summary)
        print(f"DATASET BUILD/PUBLISH FAILED CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
