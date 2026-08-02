# Optimizer Strategy

_Last updated: 2026-08-02_

## Decision

On 2026-08-02 the user selected a **hybrid Muon + AdamW strategy** as the optimizer-family candidate to implement and compare against the existing pure-AdamW baseline.

The split is by **parameter role**, not merely by layer name or tensor rank. A single block may contain parameters handled by both optimizers.

The user then approved the initial trainer-policy direction:

- source-anchor the first Muon implementation to the disclosed Kimi K3 and DeepSeek-V4 recipes rather than independently deriving a new variant;
- use DeepSeek-V4's whole-matrix Muon formulation as the first concrete implementation reference;
- keep Kimi K3's per-head Q/K/V Muon as a later ablation;
- use a shared token-count WSD schedule, with an independently configurable Muon effective-LR/update-scale multiplier;
- begin with a low effective token batch and increase it only after stability and memory measurements;
- use global gradient clipping at `1.0` initially;
- use FP16 model execution with `GradScaler`, while keeping Muon momentum and Newton-Schulz arithmetic in FP32 for qualification;
- use weight decay `0.1` for both optimizer branches, retaining the existing no-decay exceptions;
- retain deterministic seed `17` for the first paired comparisons;
- defer exact token budgets, evaluation cadence, and final model-selection policy until the hybrid trainer path is wired and qualified.

This is a deliberate **source-anchored engineering choice**. The project did not independently derive or exhaustively compare Muon variants before adopting the initial recipe. The record below preserves the external rationale and the places where this project intentionally departs from the frontier-scale papers.

The optimizer-family direction, initial routing policy, WSD scheduler family, clipping default, precision boundary, weight decay, and seed are now selected. Peak learning rates, warmup/stable/decay horizons, effective token batch stages, final Muon update scaling, and final optimizer choice remain experimental.

## Paper anchors

### Kimi K3

Source: Kimi Team, *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653, sections 2.5 and 3.2-3.3.

Short retained excerpts:

> "Kimi K3 adopts Muon as the optimizer for its matrix parameters."

> "We use a cosine learning rate schedule with a 1% linear warmup."

Kimi K3 partitions Q, K, and V momentum matrices by attention head before Newton-Schulz orthogonalization. It reports that per-head treatment balances update scales across heads and slightly reduces optimizer overhead. Kimi also reports that independently tuned cosine schedules beat independently tuned WSD schedules in its own scaling-law study; it warns that sharing peak learning rate and batch hyperparameters across scheduler families is an unfair comparison.

Project interpretation: retain Kimi's per-head treatment as a later controlled ablation. Do not use it in the first hybrid qualification because it changes both optimizer routing granularity and numerical behavior at once. The project deliberately chooses WSD for continuation flexibility and will tune it on its own scale rather than claiming to reproduce Kimi's cosine result.

### DeepSeek-V4

Source: DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348, sections 2.4 and 4.2.2.

Short retained excerpts:

> "All other modules are updated with Muon."

> "use hybrid Newton-Schulz iterations for orthogonalization."

> "rescale the RMS of each update matrix to 0.18"

DeepSeek-V4 keeps embeddings, the prediction head, RMSNorm weights, and selected static biases/gates on AdamW, while applying Muon to the remaining logically independent matrices. Its disclosed Muon recipe uses Nesterov momentum `0.95`, decoupled weight decay `0.1`, ten hybrid Newton-Schulz iterations, and target update RMS `0.18` so the AdamW learning-rate schedule can be reused. The ten iterations use eight aggressive steps with coefficients `(3.4445, -4.7750, 2.0315)`, followed by two stabilizing steps with `(2.0, -1.5, 0.5)`.

Project interpretation: use this as the first exact whole-matrix Muon reference, but run its momentum and orthogonalization in FP32 during qualification. This is not a literal reproduction of DeepSeek's trillion-parameter BF16 system; it is a numerically conservative T4 adaptation.

### Post-2025 sub-1B context

Source: Wen et al., *Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization*, arXiv:2606.16899.

The 2026 Hyperball work evaluates Qwen3-style models up to 1.2B parameters and reports that constraining weight and update Frobenius norms can improve Muon over ordinary decoupled-weight-decay baselines. This is especially relevant to the project's approximately-20M and approximately-100M regime.

Project interpretation: Hyperball is a meaningful later candidate, not part of the first implementation. First qualify conventional hybrid Muon + AdamW so any later hypersphere/norm-constraint improvement has a clean baseline.

## Initial routing policy

### Muon

Use Muon for ordinary two-dimensional learned transformations between feature spaces:

- SwiGLU `gate.weight`, `up.weight`, and `down.weight`;
- MHA `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `gate_proj.weight`, and `out_proj.weight`;
- GDN-2 dense projection weights: `q_proj`, `k_proj`, `v_proj`, `erase_proj`, `write_proj`, both `decay_proj` matrices, both `output_gate` matrices, and `out_proj`.

The first implementation orthogonalizes each complete logical matrix. It must not fuse unrelated matrices or split Q/K/V per head initially. Per-head Muon remains a later ablation.

### AdamW

Keep AdamW for parameters that are not ordinary dense feature-transform matrices:

- the tied token embedding / LM-head matrix;
- every RMSNorm scale;
- all biases;
- GDN-2 `A_log` and `dt_bias` dynamics;
- GDN-2 depthwise convolution kernels `q_conv`, `k_conv`, and `v_conv`;
- any scalar, vector, structured temporal filter, or newly added parameter not explicitly admitted to Muon.

Existing no-weight-decay exclusions remain in force for normalization scales, `A_log`, `dt_bias`, and the output-gate bias. The tied embedding remains in AdamW's decayed group unless later evidence supports an additional exclusion.

## First Muon implementation reference

The first qualification implementation should use:

```text
formulation: whole-matrix Muon
momentum: Nesterov
momentum coefficient: 0.95
matrix normalization: Frobenius normalization before Newton-Schulz
Newton-Schulz: DeepSeek-V4 ten-step hybrid polynomial
iterations 1-8: (3.4445, -4.7750, 2.0315)
iterations 9-10: (2.0, -1.5, 0.5)
target update RMS: 0.18 qualification default, configurable
Muon weight decay: 0.1
Muon momentum state: FP32
Newton-Schulz arithmetic: FP32 during qualification
per-head Q/K/V: disabled initially
```

The target update RMS and Muon LR multiplier remain configurable and must be screened. The first implementation should reproduce the disclosed mechanics, not assume that DeepSeek's optimal effective update size transfers exactly to a 20M or 100M dense hybrid.

## Scheduler and learning-rate policy

Use one shared **token-count WSD schedule** for both optimizer branches:

```text
AdamW effective LR = scheduled LR
Muon effective LR = scheduled LR × configurable Muon multiplier
```

The scheduler advances once per successful atomic update according to committed target-token count. Warmup, stable, and decay horizons remain unset until bounded runs establish the effective batch and token budget. The minimum-LR ratio also remains experimental.

WSD is a deliberate project choice rather than a claim that it matches Kimi K3. Kimi selected cosine after independently tuning both schedules; DeepSeek-V4 used linear warmup, a long stable plateau, and a final cosine decay. Our WSD implementation preserves a similar warmup/stable/final-decay structure while allowing continuation and horizon changes.

## Batch-growth policy

Begin at a low effective target-token batch and increase it only after the current level passes:

- finite-loss and finite-gradient checks;
- stable FP16 scaler behavior;
- acceptable clipping frequency;
- deterministic interruption/resume;
- T4 memory and throughput measurement;
- absence of data starvation.

The exact initial batch and growth stages are not frozen. Microbatch size may remain one sequence while gradient accumulation raises the effective target-token batch.

## Gradient clipping

Use one global L2 gradient clip with initial threshold:

```text
max_grad_norm = 1.0
```

The order is:

1. backward pass;
2. FP16 unscale;
3. measure global and per-branch gradient norms;
4. apply one global clip;
5. perform Muon and AdamW updates;
6. advance the scheduler once;
7. acknowledge the prepared block.

Log pre-clip global norm, post-clip norm, clipping frequency, per-branch gradient norms, and update-to-weight RMS.

## Precision policy

Initial qualification boundary:

```text
model forward/backward: FP16
loss scaling: CUDA GradScaler
GDN sensitive recurrence/chunkwise internals: FP32 as already implemented
Muon momentum state: FP32
Muon Newton-Schulz arithmetic: FP32
AdamW state: existing trainer behavior
```

Reduced-precision Newton-Schulz may be tested only after FP32 qualification establishes a reference trajectory and numerical bounds.

## Weight decay

Use qualification default:

```text
Muon weight decay: 0.1
AdamW weight decay: 0.1
```

Retain the model's explicit no-decay exclusions. Do not add special decay exceptions without measured evidence.

## Seed and comparison policy

Use deterministic seed `17`, matching the current trainer default, for the first paired AdamW-versus-hybrid comparisons. This avoids gratuitous configuration drift.

One seed is sufficient for software and stability screening. Any optimizer result used to select the approximately-100M substantive recipe should later be repeated across multiple seeds, with paired runs sharing initialization, data order, token batch, scheduler shape, and token budget.

## Training-step contract

One model loss and one backward pass produce gradients for both parameter sets. After FP16 unscaling and global gradient clipping:

1. Muon updates its matrix parameters;
2. AdamW updates its exception parameters;
3. both updates count as one atomic optimizer step;
4. a prepared data block is acknowledged only after both updates succeed.

Checkpoint state must include both optimizers' complete states, routing identity, Muon orthogonalization configuration, update scaling, and scheduler state. Resume must reject parameter-routing drift.

## Implementation safeguards

- Classify parameters explicitly by module role and exact name patterns, not only `parameter.ndim == 2`.
- Require every trainable parameter to belong to exactly one optimizer branch.
- Fail on overlap or unclassified parameters.
- Preserve complete logical matrices for Muon.
- Report parameter names and counts for Muon, AdamW-decay, and AdamW-no-decay groups before training.
- Keep the existing pure-AdamW path as the mandatory control.
- Store the source recipe identity and all deviations in checkpoints.
- Begin with whole-matrix Muon; Per-Head Muon, Hyperball, and SOAP/KL-SOAP remain later experiments.

## Experimental sequence

1. Wire and test the composite Muon + AdamW optimizer and exact routing audit.
2. Qualify pure AdamW under the integrated schema-v2 trainer.
3. Qualify hybrid Muon + AdamW with identical data order, initialization, token batch, clipping, and WSD shape.
4. Begin with a low effective token batch and increase only after each stage passes stability and resume gates.
5. Tune AdamW peak LR, Muon LR multiplier/update scale, and WSD horizons separately rather than copying frontier values blindly.
6. Compare loss per token, wall-clock throughput, peak memory, gradient statistics, clipping frequency, scaler behavior, optimizer-state cost, and interruption/resume determinism.
7. Decide token budgets, evaluation cadence, and model-selection policy after the hybrid path is operational.
8. Only after the base hybrid is stable, test per-head Q/K/V Muon, Hyperball, or narrower routing ablations.
