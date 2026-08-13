---
status: accepted
date: 2026-08-13
supersedes: null
---

# 0063 — Document local chat CLI usage in source

Decision: keep a compact comment header at the beginning of `chat.py` showing a valid invocation plus the meanings of `--model_params` and `--num_tokens`. Keep detailed behavior in `--help` and the implementation.

Validation: the header must stay aligned with `_parse_args`.
