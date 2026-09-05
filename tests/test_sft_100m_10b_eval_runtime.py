from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_SRC = ROOT / "kaggle" / "src"
if str(KAGGLE_SRC) not in sys.path:
    sys.path.insert(0, str(KAGGLE_SRC))

import sft_100m_10b_eval  # noqa: E402


class SFT100M10BEvalRuntimeTests(unittest.TestCase):
    def test_eval_runner_uses_uv_environment_python(self) -> None:
        bundle = Path("/tmp/small-llm-10b-sft-bundle")
        captured: dict[str, object] = {}
        profile = SimpleNamespace(
            parent_run_id="100m-10b-deep-decay-from-step15500",
            sft_run_id="100m-10b-sft-s0-2b10pct-data-001",
            parent_pointer="latest",
            run_root=Path("/tmp/small-llm-10b-sft-run"),
        )

        def capture_run(command, *, cwd):
            captured["command"] = list(command)
            captured["cwd"] = cwd
            return 0

        with (
            mock.patch.object(sft_100m_10b_eval.base, "_find_bundle", return_value=bundle),
            mock.patch.object(
                sft_100m_10b_eval.sft_scaled_runtime,
                "_verify_published_10pct_training_bundle",
            ),
            mock.patch.object(sft_100m_10b_eval.base, "_run", side_effect=capture_run),
        ):
            result = sft_100m_10b_eval.evaluate(
                profile,
                dataset_dir=str(bundle),
                eval_dir="/tmp/eval-core",
                parent_repo_id="owner/models",
                checkpoint_repo_id="owner/models",
                output="/tmp/result.json",
                device="cuda",
                precision="fp16",
                batch_size=4,
            )

        self.assertEqual(result, 0)
        command = captured["command"]
        assert isinstance(command, list)
        prefix = sft_100m_10b_eval.base._uv_prefix()
        self.assertEqual(command[: len(prefix)], prefix)
        self.assertEqual(command[len(prefix)], "python")
        self.assertNotEqual(command[len(prefix)], sys.executable)
        self.assertIn("--parent-pointer", command)
        self.assertIn("latest", command)
        self.assertEqual(captured["cwd"], sft_100m_10b_eval.REPO)


if __name__ == "__main__":
    unittest.main()
