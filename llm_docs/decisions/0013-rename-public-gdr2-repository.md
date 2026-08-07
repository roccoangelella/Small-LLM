---
status: accepted
date: 2026-08-07
supersedes: 0012
---

# 0013 — Rename the public Gated Delta Rule-2 repository

## Context

ADR 0012 bound the standalone public Gated Delta Rule-2 implementation to the GitHub repository `roccoangelella/Gated-Delta-Nets-2` while keeping the Python distribution/import identity `gated-delta-rule` / `gated_delta_rule`.

The repository name `Gated-Delta-Nets-2` could be read as a full or official Gated DeltaNet-2 implementation, while the project is intentionally centered on the independently authored reusable Gated Delta Rule-2 primitive.

## Decision

Rename the canonical public GitHub repository to:

```text
https://github.com/roccoangelella/Gated-Delta-Rule-2
```

Keep the Python identities unchanged:

```text
gated-delta-rule
gated_delta_rule
```

The public documentation and package metadata must use the renamed repository as the canonical source URL while preserving the existing non-affiliation, clean-room provenance, NVIDIA-source exclusion, MIT-license, and experimental-high-level-block constraints from ADR 0011.

## Consequences

- `Gated-Delta-Rule-2` is the canonical repository identity.
- The name now matches the stable public abstraction more precisely and reduces the chance of implying that the project is NVIDIA's official Gated DeltaNet-2 implementation.
- GitHub's redirect from the previous repository URL may continue to work, but internal documentation and package metadata should not rely on that redirect.
- ADR 0012 is superseded only with respect to the repository name; its rationale for separating the GitHub repository identity from the Python package/import identity remains historical context.

## Links

- [`0011-publish-standalone-gated-delta-rule-package.md`](0011-publish-standalone-gated-delta-rule-package.md)
- [`0012-bind-public-gdr2-repository-identity.md`](0012-bind-public-gdr2-repository-identity.md)
- Public repository: https://github.com/roccoangelella/Gated-Delta-Rule-2
