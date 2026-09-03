#!/usr/bin/env python3
"""Fast, evidence-preserving W&B startup preflight for Kaggle launchers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import ssl
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

EXPECTED_PYTHON = (3, 13)
EXPECTED_WANDB_VERSION = "0.26.1"
DELETED_RUN_MARKER = "was previously created and deleted"
PRESERVED_DEBUG_LOGS = ("debug.log", "debug-internal.log", "debug-core.log")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_debug_logs(root: Path) -> str | None:
    """Return a stable failure classification without exposing credentials."""

    for path in root.rglob("debug-internal.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if DELETED_RUN_MARKER in text:
            return "deleted_run_id"
    return None


def preserve_debug_logs(root: Path) -> dict[str, dict[str, object]]:
    """Copy the newest online-run debug logs to deterministic evidence paths."""

    preserved = root / "preserved"
    preserved.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, object]] = {}
    for name in PRESERVED_DEBUG_LOGS:
        candidates = [
            path
            for path in root.rglob(name)
            if preserved not in path.parents and path.is_file()
        ]
        if not candidates:
            continue
        source = max(
            candidates,
            key=lambda path: (
                int("online" in path.parts),
                path.stat().st_mtime_ns,
            ),
        )
        target = preserved / name
        shutil.copy2(source, target)
        result[name] = {
            "path": str(target),
            "source": str(source),
            "sha256": sha256(target),
            "byte_size": target.stat().st_size,
        }
    return result


def _phase(
    rows: list[dict[str, object]],
    name: str,
    action: Callable[[], object],
) -> object:
    started = time.perf_counter()
    try:
        value = action()
    except BaseException as error:
        rows.append(
            {
                "name": name,
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    rows.append(
        {
            "name": name,
            "status": "passed",
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--entity")
    parser.add_argument("--init-timeout", type=float, default=30.0)
    return parser.parse_args()


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = _arguments()
    root = args.dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    phases: list[dict[str, object]] = []
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "project": args.project,
        "entity_explicit": bool(args.entity),
        "run_id": args.run_id,
        "init_timeout_seconds": args.init_timeout,
        "phases": phases,
    }
    started = time.perf_counter()

    try:
        if sys.version_info[:2] != EXPECTED_PYTHON:
            raise RuntimeError(
                f"expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
                f"got {sys.version_info.major}.{sys.version_info.minor}"
            )
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is unavailable") from error
        if wandb.__version__ != EXPECTED_WANDB_VERSION:
            raise RuntimeError(
                f"expected wandb=={EXPECTED_WANDB_VERSION}, got {wandb.__version__}"
            )

        def require_api_key() -> bool:
            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError("WANDB_API_KEY is missing from the preflight process")
            return True

        _phase(phases, "secret_propagation", require_api_key)
        _phase(
            phases,
            "dns",
            lambda: socket.getaddrinfo(
                "api.wandb.ai", 443, type=socket.SOCK_STREAM
            ),
        )

        def tls_probe() -> str:
            context = ssl.create_default_context()
            with socket.create_connection(("api.wandb.ai", 443), timeout=10) as raw:
                with context.wrap_socket(raw, server_hostname="api.wandb.ai") as secure:
                    return secure.version()

        _phase(phases, "tls", tls_probe)
        _phase(
            phases,
            "api_key_authentication",
            lambda: wandb.login(
                key=os.environ["WANDB_API_KEY"], verify=True, relogin=True
            ),
        )

        def offline_core() -> str:
            run = wandb.init(
                project=args.project,
                mode="offline",
                dir=str(root / "offline"),
                settings=wandb.Settings(init_timeout=10),
            )
            if run is None:
                raise RuntimeError("offline wandb.init returned no run")
            run_id = str(run.id)
            run.finish()
            return run_id

        _phase(phases, "local_wandb_core", offline_core)

        def online_run() -> dict[str, str | None]:
            run = wandb.init(
                project=args.project,
                entity=args.entity,
                id=args.run_id,
                resume="allow",
                name=args.run_name,
                mode="online",
                dir=str(root / "online"),
                job_type="pretraining-qualification",
                tags=["kaggle", "wandb-preflight"],
                settings=wandb.Settings(init_timeout=args.init_timeout),
            )
            if run is None:
                raise RuntimeError("online wandb.init returned no run")
            identity = {
                "run_id": str(run.id),
                "entity": str(run.entity) if getattr(run, "entity", None) else None,
                "project": str(run.project) if getattr(run, "project", None) else None,
            }
            run.finish()
            return identity

        result["online_run"] = _phase(phases, "project_run_resume", online_run)
        result["status"] = "passed"
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        traceback.print_exc()
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["failure_classification"] = classify_debug_logs(root)
        result["debug_logs"] = preserve_debug_logs(root)
        _write_result(args.result, result)
        print(json.dumps(result, sort_keys=True), flush=True)

    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
