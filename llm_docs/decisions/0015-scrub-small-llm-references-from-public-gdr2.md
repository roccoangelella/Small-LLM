---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0015 — Scrub Small-LLM references from the public Gated Delta Rule-2 repository

## Context

The standalone public `Gated-Delta-Rule-2` repository should read as an independent software project. The user explicitly decided that the public repository must not contain references to the internal Small-LLM project.

The active public file tree was already simplified under ADR 0014, but older Git commit diffs still contain historical wording that mentioned Small-LLM.

## Decision

The public repository must contain no `Small-LLM`, `Small LLM`, or equivalent internal-project references in its active source tree, README, package metadata, tests, examples, CI, issues, or pull-request discussion.

Public provenance wording should be limited to what is independently relevant to the standalone package: the implementation is independently authored from the published Gated Delta Rule-2 mathematical specification, MIT licensed, contains no NVIDIA source code, and is not affiliated with NVIDIA or the PyTorch Foundation.

Detailed Small-LLM implementation lineage remains internal to this repository's `llm_docs/` memory and must not be reintroduced into the standalone public package.

Because historical Git commits can retain deleted text in diffs, a literal full-history scrub requires rewriting the public repository to a clean root/squashed history. Until such a history rewrite is performed, the active branch tree is considered clean but historical commit objects may still expose the removed wording.

## Verification

Search the public active tree for at least:

```text
Small-LLM
Small LLM
Small
LLM
```

All must return zero hits attributable to the internal project name.

## Canonical public repository

https://github.com/roccoangelella/Gated-Delta-Rule-2
