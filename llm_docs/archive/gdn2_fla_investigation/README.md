# GDN-2 / FLA investigation archive

This directory preserves the completed August 8, 2026 investigation that qualified the mixed `fla-core==0.5.2` CUDA backend.

Current production semantics belong in [`../../reference/gdn2_fla_backend.md`](../../reference/gdn2_fla_backend.md). Measurements remain under `../../evidence/`. Durable choices remain in ADRs 0016–0021.

The canonical historical narrative retained here is `gdn2_fla_investigation_handoff.md`.

Three redundant snapshots that previously lived in `llm_docs/current/` were retired during the 2026-08-10 memory cleanup:

- `gdn2_fla_amp_blocker.md`
- `gdn2_fla_qualification.md`
- `gdn2_fla_fp32_qualification.md`

Their exact pre-cleanup contents remain recoverable from Git history at commit `4c711ed88cc6554b9d1adab045deba082e171ec9`. They are intentionally not duplicated here because the canonical handoff, current reference, evidence, and ADRs already preserve the useful information at the correct semantic layers.
