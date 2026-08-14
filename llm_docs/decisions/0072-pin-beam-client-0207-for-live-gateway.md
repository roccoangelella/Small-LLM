---
status: accepted
date: 2026-08-14
supersedes: 0068
---

# 0072 — Pin Beam client 0.2.207 for the live gateway

## Context and problem statement

ADR 0068 made plain `uv sync` the complete runtime and pinned
`beam-client==0.2.201`. On 2026-08-14 the live Beam gateway rejected that client
and required at least 0.2.202. A separately provisioned `beam-client==0.2.207`
environment successfully authenticated and listed the production Beam Volume.
The canonical environment therefore cannot launch until its exact pin moves to
a gateway-compatible release.

## Considered options

- Keep 0.2.201 and launch from an untracked temporary environment.
- Pin the minimum reported 0.2.202 release.
- Pin the already-live-verified 0.2.207 release.
- Use an open-ended Beam client range.

## Decision outcome

Chosen option: **retain plain `uv sync` as the complete canonical setup and pin
`beam-client==0.2.207` in both the base dependency and compatibility extra.**

Regenerate and commit `uv.lock`. Do not launch the canonical training run from
the temporary environment, and do not loosen the pin to a moving range.

## Consequences

### Positive

- The committed environment satisfies the live gateway and matches the client
  already used to verify the production Beam Volume.
- Launch and resume remain reproducible from one locked project environment.

### Negative or limiting

- The provider can enforce another minimum later, requiring another explicit
  pin and lock update.
- This is an operational compatibility migration; it does not qualify changes
  in remote GPU runtime behavior by itself.

## Validation

- `uv sync --locked` succeeds from the committed lock.
- the installed distribution reports `beam-client 0.2.207`;
- read-only Beam gateway operations succeed;
- the canonical Beam dry run resolves the fresh 100M/10B launch contract.

## Links

- [`0068-make-plain-uv-sync-install-complete-runtime.md`](0068-make-plain-uv-sync-install-complete-runtime.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)

