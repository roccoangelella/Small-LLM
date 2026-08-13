"""Regression guards for Beam remote handler module resolution."""

from __future__ import annotations

import ast
import importlib.machinery
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEAM = ROOT / "beam"
BRIDGE = BEAM / "__init__.py"
LAUNCH = BEAM / "launch.py"


def test_repo_root_resolves_beam_launch_as_project_module() -> None:
    package = importlib.machinery.PathFinder.find_spec("beam", [str(ROOT)])
    assert package is not None
    assert package.origin == str(BRIDGE)
    assert package.submodule_search_locations is not None

    launch = importlib.machinery.PathFinder.find_spec(
        "beam.launch", list(package.submodule_search_locations)
    )
    assert launch is not None
    assert launch.origin == str(LAUNCH)


def test_beam_namespace_bridge_forwards_launcher_sdk_symbols_to_beta9() -> None:
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"), filename=str(BRIDGE))
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "beta9"
        for alias in node.names
    }
    assert {"Image", "Volume", "function"} <= imports
