# Optimizer Strategy

_Last updated: 2026-08-03_

## Decision

The project uses a **hybrid whole-matrix Muon + AdamW optimizer** for the first integrated approximately-20M GDN-2 run.

This is no longer only a candidate to implement. On 2026-08-03 the user explicitly confirmed that the desired first-run optimizer is the Muon + AdamW architecture already described in this document. Pure AdamW remains the mandatory matched control, but it is not the default launch optimizer.

The split is by parameter role, not only by tensor rank or layer name. A single decoder block contains parameters handled by both branches.

The selected initial direction remains:

- source-anchor Muon to the disclosed Kimi K3 and DeepSeek-V4 recipes rather than inventing a new variant;
- use DeepSeek-V4-style whole-matrix Muon first;
- keep Kimi K3-style per-head Q/K/V Muon as a later ablation;
- use one token-count WSD schedule with a configurable Muon LR multiplier;
- start with a low effective token batch and increase only after stability and memory evidence;
- use one global gradient clip at `1.0`;
- use FP16 model execution with `GradScaler`;
- keep Muon momentum and Newton-Schulz arithmetic in FP32 for qualification;
- use weight decay `0.1` for both branches, retaining the existing no-decay exceptions;
- use deterministic seed `17` for the first paired comparisons.

The optimizer family, routing, Newton-Schulz recipe, clipping default, precision boundary, weight decay, and seed are selected. Peak learning rate, Muon LR multiplier, token batch, WSD horizons, evaluation cadence, and token budget remain experimental and are listed in `20m_training_readiness.md`.

## Paper anchors

### Kimi K3

Source: Kimi Team, *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653, sections 2.5 and 3.2–3.3.

The retained project interpretation is:

- Kimi applies Muon to matrix parameters;
- it partitions Q, K, and V momentum matrices by attention head before orthogonalization;
- per-head treatment is a later controlled ablation here;
- Kimi selected cosine after independently tuning cosine and WSD, so its scheduler result must not be copied without matched tuning.

We deliberately do not begin with per-head Q/K/V because that would change routing granularity and numerical behavior at the same time as introducing Muon.

### DeepSeek-V4

Source: DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348, sections 2.4 and 4.2.2.

The first implementation follows the disclosed whole-matrix mechanics:

```text
Nesterov momentum: 0.95
Frobenius normalization before Newton-Schulz
iterations 1–8: (3.4445, -4.7750, 2.0315)
iterations 9–10: (2.0, -1.5, 0.5)
qualification target update RMS: 0.18
decoupled Muon weight decay: 0.1
```

The project keeps momentum and orthogonalization in FP32. This is a conservative T4 adaptation, not a literal reproduction of a frontier BF16 training stack.

### Post-2025 sub-1B context

Source: Wen et al., *Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization*, arXiv:2606.16899.

Hyperball is relevant to the project's approximately-20M and approximately-100M regime because it evaluates Qwen3-style models up to 1.2B parameters. It remains a later candidate. First we need a clean conventional hybrid Muon + AdamW baseline.

## Implemented optimizer boundary

`trainer/optimizer.py` now contains:

- the existing pure-AdamW control;
- explicit fail-closed parameter routing;
- a single atomic `HybridMuonAdamW` optimizer;
- whole-matrix FP32 Newton-Schulz orthogonalization;
- FP32 Nesterov momentum state for Muon;
- FP32 first- and second-moment state for AdamW;
- one optimizer state dictionary covering both branches;
- recipe and routing identity embedded in checkpoint state;
- resume rejection when recipe or routing changes;
- scheduler-compatible per-group LR multipliers.

The bounded trainer CLI default is now:

```text
optimizer = hybrid_muon_adamw
```

The control remains available explicitly:

```text
--optimizer adamw
```

## Initial routing policy

### Muon

Muon receives complete ordinary two-dimensional feature-transform matrices:

- SwiGLU `gate.weight`, `up.weight`, and `down.weight`;
- MHA `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `gate_proj.weight`, and `out_proj.weight`;
- GDN-2 `q_proj.weight`, `k_proj.weight`, `v_proj.weight`;
- GDN-2 `erase_proj.weight` and `write_proj.weight`;
- both GDN-2 `decay_proj` matrices;
- both GDN-2 `output_gate` matrices;
- GDN-2 `out_proj.weight`.

The first implementation orthogonalizes each complete logical matrix. It does not fuse unrelated matrices and does not split Q/K/V by head.

### AdamW

AdamW receives parameters that are not ordinary feature-transform matrices:

- the tied token embedding / LM-head matrix;
- every RMSNorm scale;
- every bias;
- GDN-2 `A_log` and `dt_bias`;
- GDN-2 depthwise Q/K/V convolution kernels;
- any scalar, vector, structured temporal filter, or newly added parameter not explicitly admitted to Muon.

The existing no-weight-decay exclusions remain:

- normalization scales;
- `A_log`;
- `dt_bias`;
- the final output-gate bias.

The tied embedding remains in AdamW's decayed group unless later evidence supports a separate exclusion.

## Fail-closed routing contract

Routing is explicit. It does not use a permissive rule such as `parameter.ndim == 2`.

Optimizer construction must prove that:

1. every trainable parameter appears exactly once;
2. no parameter appears in both branches;
3. every Muon parameter is a complete rank-2 logical matrix;
4. every named no-decay parameter exists;
5. any unrecognized trainable parameter aborts construction.

This matters because silently routing a new structured parameter to Muon could change the optimizer architecture without a decision or checkpoint-identity change.

## Whole-matrix Muon update

For each Muon matrix:

1. convert the gradient to FP32;
2. update FP32 Nesterov momentum with coefficient `0.95`;
3. form the Nesterov direction;
4. normalize by the Frobenius norm;
5. transpose temporarily when needed so Newton-Schulz operates on the smaller Gram matrix;
6. apply eight aggressive and two stabilizing polynomial iterations;
7. transpose back;
8. rescale the resulting update to the configured target RMS;
9. apply decoupled weight decay;
10. apply the scheduled Muon learning rate.

The qualification target RMS is `0.18`, but it remains configurable because transfer from frontier-scale training to 20M/100M models must be measured.

A zero gradient produces a zero Muon update. Non-finite gradients or Newton-Schulz results fail loudly.

## AdamW update

The AdamW branch keeps the selected baseline values:

```text
betas: (0.9, 0.95)
epsilon: 1e-8
weight decay: 0.1 for the decay group
state arithmetic: FP32
```

Both branches update inside one optimizer `step()`. A prepared block is acknowledged only after the complete hybrid step succeeds.

## Scheduler and learning-rate policy

Both branches share one base token-count schedule:

```text
AdamW effective LR = scheduled base LR
Muon effective LR = scheduled base LR × muon_lr_multiplier
```

`TokenLRScheduler` now preserves the per-group multiplier instead of overwriting every group with the same LR.

WSD remains the selected scheduler family for continuation flexibility. This does not claim that WSD is universally better than cosine. The base LR, Muon multiplier, warmup, stable horizon, decay horizon, and floor still need bounded tuning.

## Batch-growth policy

Begin at a low effective target-token batch and increase it only after the current level passes:

- finite loss and gradients;
- stable scaler behavior;
- acceptable clipping frequency;
- deterministic interruption/resume;
- T4 memory and throughput measurement;
- absence of data starvation.

The accepted dataset-operation pilot used 512 sequences per block, or about 1.05M target tokens per optimizer update at context 2,048. That cache is valid operational evidence but is not suitable for this batch-growth policy. The block-size explanation and unresolved training-cache choice are in `20m_training_readiness.md`.

## Gradient clipping

Use one global L2 clip with initial threshold:

```text
max_grad_norm = 1.0
```

The intended order is:

1. backward pass;
2. FP16 unscale;
3. measure global and per-branch gradient norms;
4. apply one global clip;
5. perform the Muon and AdamW updates atomically;
6. advance the scheduler once;
7. acknowledge the prepared block.

The current trainer performs the global clip and atomic step. Per-branch gradient and update statistics remain instrumentation work before the longer qualification segment.

## Precision policy

Initial qualification boundary:

```text
model forward/backward: FP16
loss scaling: CUDA GradScaler
GDN sensitive recurrence/chunkwise internals: FP32
Muon momentum state: FP32
Muon Newton-Schulz arithmetic: FP32
AdamW moment state: FP32
```

Reduced-precision Newton-Schulz is not authorized before FP32 qualification establishes a stable reference.

## Checkpoint identity

A hybrid checkpoint contains:

- complete model state;
- complete Muon and AdamW optimizer state;
- parameter-group and parameter-order state;
- the exact recipe identifier;
- Newton-Schulz coefficients and iteration counts;
- momentum, target RMS, Muon LR multiplier, and weight decay;
- the exact routed parameter-name lists;
- scheduler and scaler state.

Resume rejects a different trainer configuration through the normal trainer identity and rejects a different hybrid recipe or routing through optimizer metadata.

## Comparison policy

The first useful pair is:

1. hybrid Muon + AdamW;
2. pure AdamW control.

The runs must share:

- initialization;
- data order;
- prepared-block geometry;
- effective token batch;
- clipping;
- schedule shape;
- token budget;
- seed.

One seed is enough for software and stability qualification. A result used to select the approximately-100M substantive recipe should later be repeated across multiple seeds.

## Implementation safeguards

- Keep the pure-AdamW path.
- Classify by exact role and name pattern.
- Fail on overlap, omission, or an unknown trainable parameter.
- Preserve complete logical matrices for Muon.
- Keep Muon state and Newton-Schulz in FP32 during qualification.
- Store recipe and routing identity in checkpoints.
- Keep per-head Q/K/V, Hyperball, SOAP, and KL-SOAP as later experiments.
- Do not silently change the optimizer family on resume.

## Experimental sequence

1. Run the complete CPU suite for routing, state, scheduler multipliers, and checkpoint round-trip.
2. Build the smaller-block training-qualification cache after its geometry is selected.
3. Run the integrated approximately-20M hybrid optimizer preflight.
4. Qualify interruption and local resume.
5. Qualify remote publication and empty-environment continuation.
6. Run the matched pure-AdamW control with the same data and schedule.
7. Tune base LR, Muon multiplier/update RMS, WSD horizons, and effective token batch separately.
8. Compare loss per token, throughput, memory, clipping, scaler behavior, optimizer-state cost, and resume behavior.
9. Only after the base hybrid is stable, test per-head Q/K/V Muon, Hyperball, or narrower routing ablations.

## Open values

The remaining engineering and experimental choices are centralized in `20m_training_readiness.md`. In particular, this document does not freeze:

- training-cache block size;
- base learning rate;
- Muon LR multiplier;
- whether target update RMS remains `0.18`;
- WSD horizons and floor;
- effective token batch stages;
- checkpoint/evaluation cadence;
- bounded token budget;
- final model-selection rule.
