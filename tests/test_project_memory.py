"""Repository contracts for the Markdown project-memory layout."""
from __future__ import annotations

import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "llm_docs"

INDEXES = (
    DOCS / "README.md",
    DOCS / "decisions" / "README.md",
    DOCS / "reference" / "README.md",
    DOCS / "runbooks" / "README.md",
    DOCS / "research" / "README.md",
    DOCS / "evidence" / "README.md",
)

REQUIRED_PATHS = (
    ROOT / "AGENTS.md",
    DOCS / "current" / "status.md",
    DOCS / "current" / "roadmap.md",
    DOCS / "decisions" / "template.md",
    DOCS / "research" / "project_memory_research.md",
    DOCS / "archive" / "README.md",
)

REMOVED_PATHS = (
    ROOT / "dataset" / "legacy",
    ROOT / "20M_training.py",
    ROOT / "kaggle" / "run_20m_full_training.py",
    ROOT / "kaggle" / "run_20m_from_clone.py",
    ROOT / "kaggle" / "run_20m_repeatability_from_clone.py",
    ROOT / "kaggle" / "run_20m_local_resume_from_clone.py",
    ROOT / "kaggle" / "run_20m_remote_recovery_from_clone.py",
    ROOT / "llm_test_trace.json",
)


def local_markdown_links(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    links: list[str] = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0]
        if target and "://" not in target and target.endswith(".md"):
            links.append(target)
    return tuple(links)


class ProjectMemoryLayoutTests(unittest.TestCase):
    def test_required_memory_files_exist(self) -> None:
        for path in REQUIRED_PATHS:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"missing project-memory file: {path}")

    def test_llm_docs_root_is_only_the_map(self) -> None:
        markdown_files = sorted(path.name for path in DOCS.glob("*.md"))
        self.assertEqual(markdown_files, ["README.md"])

    def test_index_relative_markdown_links_resolve(self) -> None:
        docs_root = DOCS.resolve()
        for index in INDEXES:
            with self.subTest(index=index):
                self.assertTrue(index.is_file())
                for target in local_markdown_links(index):
                    resolved = (index.parent / target).resolve()
                    self.assertTrue(
                        resolved.is_relative_to(docs_root),
                        f"index link escapes llm_docs: {index} -> {target}",
                    )
                    self.assertTrue(
                        resolved.is_file(),
                        f"broken index link: {index} -> {target}",
                    )

    def test_adrs_use_numbered_single_decision_shape(self) -> None:
        decisions = DOCS / "decisions"
        adrs = sorted(decisions.glob("[0-9][0-9][0-9][0-9]-*.md"))
        self.assertGreaterEqual(len(adrs), 3)
        required_headings = (
            "## Context and problem statement",
            "## Considered options",
            "## Decision outcome",
            "## Consequences",
        )
        for adr in adrs:
            text = adr.read_text(encoding="utf-8")
            with self.subTest(adr=adr):
                self.assertTrue(text.startswith("---\n"), "ADR needs YAML metadata")
                for heading in required_headings:
                    self.assertIn(heading, text)

    def test_agent_map_stays_small_and_points_to_current_memory(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 100)
        self.assertIn("llm_docs/current/status.md", text)
        self.assertIn("llm_docs/current/roadmap.md", text)
        self.assertIn("llm_docs/decisions/README.md", text)

    def test_removed_legacy_paths_do_not_return(self) -> None:
        for path in REMOVED_PATHS:
            with self.subTest(path=path):
                self.assertFalse(path.exists(), f"removed legacy path returned: {path}")


if __name__ == "__main__":
    unittest.main()
