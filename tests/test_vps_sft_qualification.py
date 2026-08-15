from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.qualification import sft_100m_2b_vps as vps


class VPSSFTQualificationTests(unittest.TestCase):
    def test_default_datasets_live_under_tests_test_datasets(self) -> None:
        self.assertEqual(vps.DEFAULT_EVAL_CORE, vps.TEST_DATA_ROOT / "eval_core_v1")
        self.assertEqual(
            vps.DEFAULT_SFT_BUNDLE,
            vps.TEST_DATA_ROOT / "100m-2b-sft-s0-001",
        )

    def test_dotenv_loads_literals_without_overwriting_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dotenv = Path(temporary) / ".env"
            dotenv.write_text(
                "# comment\n"
                "SMALL_LLM_SFT_HF_REPO_ID=owner/from-file\n"
                "HF_TOKEN='file-token'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HF_TOKEN": "existing-token"}, clear=True):
                vps._load_dotenv(dotenv)
                self.assertEqual(
                    os.environ["SMALL_LLM_SFT_HF_REPO_ID"],
                    "owner/from-file",
                )
                self.assertEqual(os.environ["HF_TOKEN"], "existing-token")

    def test_main_uses_local_datasets_and_provider_neutral_eval_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            eval_dir = root / "eval"
            bundle.mkdir()
            eval_dir.mkdir()
            (bundle / "bundle-manifest.json").write_text("{}\n", encoding="utf-8")
            (eval_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            output = root / "report.json"

            with (
                patch.object(vps, "_load_dotenv"),
                patch.object(vps, "verify_bundle") as verify_bundle,
                patch.object(vps, "verify_eval_core") as verify_eval_core,
                patch.object(vps.eval_suite, "main", return_value=0) as evaluate,
            ):
                result = vps.main(
                    [
                        "--dataset-dir", str(bundle),
                        "--eval-dir", str(eval_dir),
                        "--repo-id", "owner/qualification",
                        "--output", str(output),
                        "--suite", "full",
                    ]
                )

            self.assertEqual(result, 0)
            verify_bundle.assert_called_once_with(bundle.resolve())
            verify_eval_core.assert_called_once_with(eval_dir.resolve())
            forwarded = evaluate.call_args.args[0]
            self.assertIn("--parent-run-id", forwarded)
            self.assertIn(vps.PARENT_RUN_ID, forwarded)
            self.assertIn("--sft-run-id", forwarded)
            self.assertIn(vps.SFT_RUN_ID, forwarded)
            self.assertEqual(forwarded.count("owner/qualification"), 2)
            self.assertNotIn("/kaggle/input", " ".join(forwarded))


if __name__ == "__main__":
    unittest.main()
