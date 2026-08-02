# Optimizer Strategy

_Last updated: 2026-08-02_

## Decision

On 2026-08-02 the user selected a **hybrid Muon + AdamW strategy** as the optimizer-family candidate to implement and compare against the existing pure-AdamW baseline.

The split is by **parameter role**, not merely by layer name or tensor rank. A single block may contain parameters handled by both optimizers.

This decision freezes the optimizer-family direction and initial routing policy. It does not freeze peak learning rate, Muon update scaling, momentum, weight decay, scheduler, clipping threshold, or the final optimizer used for the substantive run; those remain subject to bounded controlled experiments.

## Initial routing policy

### Muon

Use Muon for ordinary two-dimensional learned transformations between feature spaces:

- SwiGLU `gate.weight`, `up.weight`, and `down.weight`;
- MHA `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `gate_proj.weight`, and `out_proj.weight`;
- GDN-2 dense projection weights: `q_proj`, `k_proj`, `v_proj`, `erase_proj`, `write_proj`, both `decay_proj` matrices, both `output_gate` matrices, and `out_proj`.

The first implementation should orthogonalize each complete logical matrix. It should not fuse unrelated matrices or split Q/K/V per head initially. Per-head Muon remains a later ablation.

### AdamW

Keep AdamW for parameters that are not ordinary dense feature-transform matrices:

- the tied token embedding / LM-head matrix;
- every RMSNorm scale;
- all biases;
- GDN-2 `A_log` and `dt_bias` dynamics;
- GDN-2 depthwise convolution kernels `q_conv`, `k_conv`, and `v_conv`;
- any scalar, vector, structured temporal filter, or newly added parameter not explicitly admitted to Muon.

Existing no-weight-decay exclusions remain in force for normalization scales, `A_log`, `dt_bias`, and the output-gate bias. The tied embedding remains in AdamW's decayed group unless later evidence supports an additional exclusion.

## Training-step contract

One model loss and one backward pass produce gradients for both parameter sets. After FP16 unscaling and global gradient clipping:

1. Muon updates its matrix parameters;
2. AdamW updates its exception parameters;
3. both updates count as one atomic optimizer step;
4. a prepared data block is acknowledged only after both updates succeed.

The scheduler advances once per atomic step and controls both optimizer branches. The initial implementation may use one shared LR schedule with an explicit Muon LR/update-scale multiplier, but optimizer-specific peak values must remain configurable and independently tuned.

Checkpoint state must include both optimizers' complete states, routing identity, Muon orthogonalization configuration, update scaling, and scheduler state. Resume must reject parameter-routing drift.

## Implementation safeguards

- Classify parameters explicitly by module role and exact name patterns, not only `parameter.ndim == 2`.
- Require every trainable parameter to belong to exactly one optimizer branch.
- Fail on overlap or unclassified parameters.
- Preserve complete logical matrices for Muon.
- Report parameter names and counts for Muon, AdamW-decay, and AdamW-no-decay groups before training.
- Keep the existing pure-AdamW path as the mandatory control.
- Begin with whole-matrix Muon and a modest Newton-Schulz implementation; Per-Head Muon and heavier SOAP/KL-SOAP variants remain later experiments.

## Experimental sequence

1. Qualify pure AdamW under the integrated schema-v2 trainer.
2. Qualify hybrid Muon + AdamW with identical data order, initialization, token batch, clipping, and scheduler shape.
3. Tune peak LR/update scale separately rather than copying AdamW values blindly.
4. Compare loss per token, wall-clock throughput, peak memory, gradient statistics, scaler behavior, and interruption/resume determinism.
5. Only after the base hybrid is stable, test per-head Q/K/V Muon or narrower routing ablations.
