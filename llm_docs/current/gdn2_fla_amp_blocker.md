---
status: current
last_reviewed: 2026-08-08
---

# FLA GDN-2 AMP blocker — resolved by corrected qualification

This file records the disposition of the former strong-decay AMP blocker. It is **not an active production block**.

The trajectory remains clean at:

```text
checkpoint: step-00004000
last_consumed_block_id: 3999
next update: 4001
```

The original first production FLA resume did encounter a genuine mixed-dtype Triton compilation failure before update 4001. That integration issue was fixed by canonicalizing ordinary FLA compute tensors to a common low-precision dtype under AMP while keeping decay/state FP32.

The later `strong_decay_-6_amp` and forced-decay sweep failures were subsequently treated as evidence of a separate FLA backward numerical instability. The final live T4 investigation found that those comparisons used an invalid adaptive-reference contract: the FP32 reference recurrence was itself called inside CUDA FP16 autocast. On reproduced failing rows, the adaptive reference gradients were non-finite while the FLA gradients were finite. Layer initialization was also not reset per decay row.

The corrected deterministic reference disables autocast only inside the adaptive recurrence. With that correction, `fla-core==0.5.2` mixed FLA and full-FP32 FLA both pass all requested constant decays:

```text
[-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
```

Both modes also pass the complete true step-4000 / block-4000 forward and all-gradient parity gate using checkpoint GradScaler scale 256, with no optimizer step executed.

The qualified production path is mixed FLA on `fla-core==0.5.2`. See:

- `llm_docs/current/gdn2_fla_qualification.md`
- `llm_docs/current/gdn2_fla_fp32_qualification.md`
- `llm_docs/evidence/gdn2_fla_corrected_oracle_and_step4000_qualification_2026-08-08.md`
- ADR 0021

Historical failed evidence remains preserved and should be interpreted through this later correction rather than deleted.