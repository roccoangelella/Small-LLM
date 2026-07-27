# Legacy decoded-text pipeline

This directory is a historical copy of the old multi-pass curation code. It
decoded GPT-2 tokens, sampled documents, called Gemini, ran code/quality
filters, planned cluster quotas, wrote JSONL shards, and audited them.

None of it is imported by the production builder, none of its commands are
available from `dataset.main`, and its old optional dependencies are no longer
installed. Do not use it to build the corpus.

The supported path is:

```bash
uv run python -m dataset.main build
uv run python -m dataset.main build --resume
uv run python -m dataset.main verify
```
