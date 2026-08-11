"""Repository-wide contracts for the consolidated dataset package/layout."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

from dataset import config

ROOT = Path(__file__).resolve().parents[1]

RETIRED_PATHS = (
    "dataset/MIXTURE_CALIBRATION.md",
    "dataset/PRODUCTION_RUNBOOK.md",
    "dataset/PRODUCTION_SMOKE_TEST.md",
    "dataset/acceptance.py",
    "dataset/mixture.py",
    "dataset/qualification_100m.py",
    "dataset/qualification_100m_report.py",
    "dataset/qualification_20m.py",
    "dataset/qualification_20m_report.py",
    "dataset/qualification_20m_verify.py",
    "dataset/qualification_500m.py",
    "dataset/qualification_500m_report.py",
    "dataset/qualification_2b.py",
    "dataset/qualification_2b_report.py",
    "dataset/src/build.py",
    "dataset/src/checkpoint.py",
    "dataset/src/exceptions.py",
    "dataset/src/manifest.py",
    "dataset/src/mixture_calibration.py",
    "dataset/src/progress_report.py",
    "dataset/src/writer.py",
)

RETIRED_MODULES = frozenset(
    {
        "dataset.acceptance",
        "dataset.mixture",
        "dataset.qualification_100m",
        "dataset.qualification_100m_report",
        "dataset.qualification_20m",
        "dataset.qualification_20m_report",
        "dataset.qualification_20m_verify",
        "dataset.qualification_500m",
        "dataset.qualification_500m_report",
        "dataset.qualification_2b",
        "dataset.qualification_2b_report",
        "dataset.src.build",
        "dataset.src.checkpoint",
        "dataset.src.exceptions",
        "dataset.src.manifest",
        "dataset.src.mixture_calibration",
        "dataset.src.progress_report",
        "dataset.src.writer",
    }
)

ACTIVE_PYTHON_ROOTS = (
    ROOT / "dataset",
    ROOT / "kaggle",
    ROOT / "trainer",
    ROOT / "post_training",
)

ACTIVE_DOCS = (
    ROOT / "README.md",
    ROOT / "dataset" / "README.md",
    *(ROOT / "llm_docs" / "current").glob("*.md"),
    *(ROOT / "llm_docs" / "reference").glob("*.md"),
    *(ROOT / "llm_docs" / "runbooks").glob("*.md"),
)

RETIRED_COMMAND_SNIPPETS = (
    "python kaggle/run_20m_100m.py",
    "python kaggle/run_20m_500m.py",
    "python kaggle/run_20m_2b.py",
    "python -m dataset.qualification_100m",
    "python -m dataset.qualification_20m",
    "python -m dataset.qualification_500m",
    "python -m dataset.qualification_2b",
    "python -m dataset.acceptance",
    "python -m dataset.mixture",
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolved_from_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    current = _module_name(path)
    package = current if path.name == "__init__.py" else current.rpartition(".")[0]
    parts = package.split(".") if package else []
    ascend = node.level - 1
    if ascend > len(parts):
        return node.module or ""
    base = parts[: len(parts) - ascend]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(_resolved_from_module(path, node))
    return imported


def _is_retired_module(name: str) -> bool:
    return any(name == retired or name.startswith(retired + ".") for retired in RETIRED_MODULES)


class DatasetLayoutTests(unittest.TestCase):
    def test_retired_dataset_paths_do_not_return(self) -> None:
        for relative in RETIRED_PATHS:
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists(), f"retired dataset path returned: {relative}")

    def test_active_python_does_not_import_retired_dataset_modules(self) -> None:
        for root in ACTIVE_PYTHON_ROOTS:
            for path in root.rglob("*.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    retired = sorted(name for name in _imports(path) if _is_retired_module(name))
                    self.assertEqual(retired, [], f"imports retired dataset modules: {retired}")

    def test_active_python_has_no_monolithic_train_validation_bin_contract(self) -> None:
        forbidden = ("train.bin", "validation.bin")
        for root in ACTIVE_PYTHON_ROOTS:
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=path.relative_to(ROOT)):
                    self.assertFalse(
                        any(name in text for name in forbidden),
                        "active source still refers to the retired monolithic dataset layout",
                    )

    def test_retired_monolithic_path_constants_are_gone(self) -> None:
        for name in (
            "TRAIN_FILENAME",
            "VALIDATION_FILENAME",
            "PROGRESS_CSV_FILENAME",
            "PROGRESS_CSV_HEARTBEAT_SECONDS",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(config, name), f"retired dataset constant returned: {name}")

    def test_active_docs_do_not_give_retired_dataset_or_launcher_commands(self) -> None:
        for path in ACTIVE_DOCS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                stale = [command for command in RETIRED_COMMAND_SNIPPETS if command in text]
                self.assertEqual(stale, [], f"active documentation gives retired commands: {stale}")

    def test_setuptools_packages_include_dataset_subpackages(self) -> None:
        payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        includes = payload["tool"]["setuptools"]["packages"]["find"]["include"]
        self.assertIn("dataset*", includes)

    def test_direct_kaggle_launcher_resolves_dataset_package_outside_repo_cwd(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "kaggle" / "launch.py"),
                    "train",
                    "--model",
                    "20M",
                    "--tokens",
                    "2B",
                    "--dry-run",
                ],
                cwd=temporary,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["profile"], "20m-2b-data-scaling-v1")
        self.assertEqual(payload["dataset_run_id"], "20m-2b-dataset-001")


if __name__ == "__main__":
    unittest.main()
