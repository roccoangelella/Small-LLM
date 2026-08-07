"""Tests for the self-provisioning complete-evaluation entry point."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trainer.eval_entrypoint import (
    _eval_dir_from_argv,
    _with_eval_dir,
    default_eval_dir,
    ensure_eval_core,
)


class EvalEntrypointTests(unittest.TestCase):
    def test_explicit_eval_dir_is_preserved(self) -> None:
        selected = _eval_dir_from_argv(["full", "--eval-dir", "/tmp/eval-core"])
        self.assertEqual(selected, Path("/tmp/eval-core"))

    def test_missing_eval_dir_is_injected(self) -> None:
        with patch.dict(os.environ, {"SMALL_LLM_EVAL_DIR": "/tmp/frozen-eval"}):
            forwarded, selected = _with_eval_dir(["full", "--repo-id", "owner/repo"])
        self.assertEqual(selected, Path("/tmp/frozen-eval"))
        self.assertEqual(forwarded[-2:], ["--eval-dir", "/tmp/frozen-eval"])

    def test_environment_overrides_default_eval_dir(self) -> None:
        with patch.dict(os.environ, {"SMALL_LLM_EVAL_DIR": "/tmp/custom-eval"}):
            self.assertEqual(default_eval_dir(), Path("/tmp/custom-eval"))

    def test_missing_corpus_is_built_then_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "eval_core_v1"

            def fake_build(path: Path) -> None:
                self.assertEqual(path, target.resolve())
                path.mkdir(parents=True)
                (path / "manifest.json").write_text("{}\n", encoding="utf-8")

            with patch("trainer.eval_entrypoint.build_eval_core", side_effect=fake_build) as build:
                with patch("trainer.eval_entrypoint.verify_eval_core") as verify:
                    resolved = ensure_eval_core(target)
            self.assertEqual(resolved, target.resolve())
            build.assert_called_once_with(target.resolve())
            verify.assert_called_once_with(target.resolve())

    def test_existing_corpus_is_verified_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "eval_core_v1"
            target.mkdir()
            with patch("trainer.eval_entrypoint.build_eval_core") as build:
                with patch("trainer.eval_entrypoint.verify_eval_core") as verify:
                    resolved = ensure_eval_core(target)
            self.assertEqual(resolved, target.resolve())
            build.assert_not_called()
            verify.assert_called_once_with(target.resolve())


if __name__ == "__main__":
    unittest.main()
