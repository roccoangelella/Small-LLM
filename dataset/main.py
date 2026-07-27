"""Command-line entry point for the Nemotron-ClimbMix curation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Callable

from dataset import config
from dataset.src.audit import audit_selected_corpus
from dataset.src.planning import create_selection_plan
from dataset.src.review import review_samples_with_llm
from dataset.src.sampling import collect_samples_and_inventory
from dataset.src.selection import select_corpus


def configure_logging() -> None:
    """Configure concise progress output for long streaming operations."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    """Run one explicit pipeline stage."""

    parser = argparse.ArgumentParser(
        description="Build a balanced, non-code Nemotron-ClimbMix pretraining subset."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("sample", help="Stream source once: make 50 samples/cluster and eligible inventory.")
    subcommands.add_parser("review", help="Review sampled clusters with Gemini 3.6 Flash.")
    subcommands.add_parser("plan", help="Create deterministic per-cluster sampling rates from inventory and quotas.")
    select_parser = subcommands.add_parser("select", help="Stream and write the quota-balanced selected corpus.")
    select_parser.add_argument("--resume", action="store_true", help="Resume from the crash-safe selection checkpoint.")
    subcommands.add_parser("audit", help="Re-audit selected output with deterministic checks and Gemini.")
    arguments = parser.parse_args()

    configure_logging()
    config.validate_config()
    actions: dict[str, Callable[[], object]] = {
        "sample": collect_samples_and_inventory,
        "review": review_samples_with_llm,
        "plan": create_selection_plan,
        "select": lambda: select_corpus(resume=arguments.resume),
        "audit": audit_selected_corpus,
    }
    result = actions[arguments.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
