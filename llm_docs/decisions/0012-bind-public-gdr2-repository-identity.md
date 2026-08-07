---
status: accepted
date: 2026-08-07
supersedes: null
---

# 0012 — Bind the public Gated Delta Rule-2 repository identity

## Context

ADR 0011 authorized publication of the independently authored Gated Delta Rule-2 implementation as a standalone MIT-licensed project with Python package identity `gated-delta-rule` / `gated_delta_rule`.

The user created a new public GitHub repository at `roccoangelella/Gated-Delta-Nets-2` and authorized using or renaming it for the release. The connected GitHub interface can write the repository contents but does not expose repository rename operations.

## Decision

Use the existing public repository as the canonical source repository:

```text
https://github.com/roccoangelella/Gated-Delta-Nets-2
```

Keep the Python distribution/import identity as:

```text
gated-delta-rule
gated_delta_rule
```

The repository documentation must make clear that the project is an independent Gated Delta Rule-2 reference implementation, is not NVIDIA's official Gated DeltaNet-2 implementation, contains no NVIDIA source code, and is not affiliated with NVIDIA or the PyTorch Foundation.

The repository name does not change the clean-room/provenance constraints established by ADR 0011. The stable public API remains the reusable Gated Delta Rule-2 functional primitive and stateless `torch.nn.Module` wrapper; the higher-level GatedDeltaNet2 convenience block remains explicitly experimental.

## Consequences

- Public-source links, package metadata, and citation metadata point to `roccoangelella/Gated-Delta-Nets-2`.
- The package name remains neutral and suitable for potential future PyPI publication.
- A future repository rename may be performed manually if desired, but it is not required for the current public release.

## Links

- [`0011-publish-standalone-gated-delta-rule-package.md`](0011-publish-standalone-gated-delta-rule-package.md)
- Public repository: https://github.com/roccoangelella/Gated-Delta-Nets-2
