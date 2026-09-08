"""Repository contracts for the Markdown project-memory layout."""
from __future__ import annotations

import hashlib
import json
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
    DOCS / "plans" / "README.md",
    DOCS / "evidence" / "README.md",
    DOCS / "archive" / "README.md",
)

REQUIRED_PATHS = (
    ROOT / "AGENTS.md",
    DOCS / "current" / "status.md",
    DOCS / "current" / "roadmap.md",
    DOCS / "decisions" / "template.md",
    DOCS / "research" / "project_memory_research.md",
    DOCS / "research" / "agent_memory_and_documentation_2026-08-10.md",
    DOCS / "plans" / "README.md",
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

CURRENT_FILES = ("roadmap.md", "status.md")
ADR_LEGACY_BASELINE = (
    ROOT / "tests" / "fixtures" / "project_memory_adr_legacy_baseline.json"
)
STRICT_ADR_HEADINGS = (
    "## Context and problem statement",
    "## Considered options",
    "## Decision outcome",
    "## Consequences",
)


def local_markdown_links(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    links: list[str] = []
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0]
        if target and "://" not in target and target.endswith(".md"):
            links.append(target)
    return tuple(links)


def has_strict_adr_shape(text: str) -> bool:
    return text.startswith("---\n") and all(
        heading in text for heading in STRICT_ADR_HEADINGS
    )


def load_adr_legacy_baseline() -> dict[str, str]:
    baseline = json.loads(ADR_LEGACY_BASELINE.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not all(
        isinstance(filename, str) and isinstance(digest, str)
        for filename, digest in baseline.items()
    ):
        raise ValueError("ADR legacy baseline must map filenames to SHA-256 strings")
    return baseline


class ProjectMemoryLayoutTests(unittest.TestCase):
    def test_required_memory_files_exist(self) -> None:
        for path in REQUIRED_PATHS:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"missing project-memory file: {path}")

    def test_llm_docs_root_is_only_the_map(self) -> None:
        markdown_files = sorted(path.name for path in DOCS.glob("*.md"))
        self.assertEqual(markdown_files, ["README.md"])

    def test_current_is_only_high_freshness_working_memory(self) -> None:
        markdown_files = sorted(path.name for path in (DOCS / "current").glob("*.md"))
        self.assertEqual(markdown_files, list(CURRENT_FILES))

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

    def test_legacy_adr_baseline_is_explicit_and_unchanged(self) -> None:
        decisions = DOCS / "decisions"
        adrs = sorted(decisions.glob("[0-9][0-9][0-9][0-9]-*.md"))
        baseline = load_adr_legacy_baseline()
        self.assertTrue(baseline)

        nonconforming = {
            adr.name
            for adr in adrs
            if not has_strict_adr_shape(adr.read_text(encoding="utf-8"))
        }
        self.assertEqual(set(baseline), nonconforming)

        for filename, expected_digest in sorted(baseline.items()):
            path = decisions / filename
            with self.subTest(adr=filename):
                self.assertEqual(path.name, filename)
                self.assertRegex(expected_digest, r"^[0-9a-f]{64}$")
                self.assertTrue(path.is_file(), f"missing baseline ADR: {filename}")
                actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual_digest, expected_digest)
                self.assertFalse(
                    has_strict_adr_shape(path.read_text(encoding="utf-8")),
                    "conforming ADR cannot inherit a legacy exemption",
                )

    def test_conforming_existing_and_new_adrs_use_standard_shape(self) -> None:
        decisions = DOCS / "decisions"
        adrs = sorted(decisions.glob("[0-9][0-9][0-9][0-9]-*.md"))
        self.assertGreaterEqual(len(adrs), 3)
        baseline = load_adr_legacy_baseline()
        for adr in adrs:
            with self.subTest(adr=adr):
                if adr.name in baseline:
                    continue
                text = adr.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("---\n"), "ADR needs YAML metadata")
                for heading in STRICT_ADR_HEADINGS:
                    self.assertIn(heading, text)

    def test_agent_map_stays_small_and_points_to_current_memory(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 100)
        self.assertIn("llm_docs/current/status.md", text)
        self.assertIn("llm_docs/current/roadmap.md", text)
        self.assertIn("llm_docs/decisions/README.md", text)
        self.assertIn("llm_docs/plans/", text)

    def test_removed_legacy_paths_do_not_return(self) -> None:
        for path in REMOVED_PATHS:
            with self.subTest(path=path):
                self.assertFalse(path.exists(), f"removed legacy path returned: {path}")


if __name__ == "__main__":
    unittest.main()
