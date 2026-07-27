"""Obsolete decoded-text/JSONL curation pipeline.

These modules implement the old multi-pass workflow that decoded documents,
ran text/code/quality filters, sent batches to a Gemini reviewer, and wrote
JSONL shards. They are kept here for reference only and are NOT part of the
production path. Do not import them from the production code.

The current production pipeline lives in :mod:`dataset.src` and is driven by
``uv run python -m dataset.main build`` / ``verify``.
"""
