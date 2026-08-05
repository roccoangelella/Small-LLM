"""Network-free regression tests for Kaggle W&B startup diagnostics."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parents[1] / "kaggle"
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

import wandb_preflight as preflight  # noqa: E402


class KaggleWandbPreflightTests(unittest.TestCase):
    def test_deleted_run_conflict_is_classified_from_internal_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "online" / "wandb" / "run-test" / "logs" / "debug-internal.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                '{"status":409,"body":"run old-id was previously created and deleted; try a new run id"}\n',
                encoding="utf-8",
            )
            self.assertEqual(preflight.classify_debug_logs(root), "deleted_run_id")

    def test_all_required_debug_logs_are_copied_to_stable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            offline = root / "offline" / "wandb" / "run-test" / "logs"
            online = root / "online" / "wandb" / "run-test" / "logs"
            offline.mkdir(parents=True)
            online.mkdir(parents=True)
            for name in preflight.PRESERVED_DEBUG_LOGS:
                (offline / name).write_text(f"offline {name}\n", encoding="utf-8")
                (online / name).write_text(f"online {name}\n", encoding="utf-8")

            result = preflight.preserve_debug_logs(root)

            self.assertEqual(set(result), set(preflight.PRESERVED_DEBUG_LOGS))
            for name in preflight.PRESERVED_DEBUG_LOGS:
                preserved = root / "preserved" / name
                self.assertTrue(preserved.is_file())
                self.assertEqual(
                    preserved.read_text(encoding="utf-8"), f"online {name}\n"
                )
                self.assertEqual(result[name]["path"], str(preserved))


if __name__ == "__main__":
    unittest.main()
