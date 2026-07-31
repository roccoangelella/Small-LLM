# Tomorrow's TODO — July 30, 2026

Tomorrow the goal is to finish the attention-architecture investigation and turn it into an actual model decision. The 2 TB exact-mixture calibration is already running on the VPS, so model work can proceed in parallel without touching the dataset subsystem unless the run reveals a real defect.

## 1. Finish Gated DeltaNet-2

Start from where today's journal stopped and complete the Gated DeltaNet-2 study.

I want to understand:

- the exact erase, write, and decay update rules;
- what changed compared with Gated DeltaNet and Kimi Delta Attention;
- how causal training is parallelized or chunked;
- the recurrent inference state and its memory cost;
- whether the official or Flash Linear Attention kernels realistically support a T4 in FP16;
- licensing and implementation constraints;
- which parts are essential and which are optional paper optimizations.

The output should be a small implementation-oriented summary, not only a paper summary. I should be able to explain the forward pass and sketch the tensors involved without referring back to the paper.

## 2. Examine the remaining useful attention models

Keep this focused. We do not need to collect every new LLM architecture; we need enough evidence to make a decision.

Study these references:

1. **OLMo Hybrid** — especially the controlled comparison between ordinary attention and a 3:1 Gated DeltaNet/full-attention hybrid.
2. **Kimi K3** — Kimi Delta Attention and Attention Residuals. Ignore the giant MoE design for our implementation.
3. **Solar Open 2** — negative-eigenvalue Gated DeltaNet and its positional-encoding choices.
4. **Ling/Ring 2.6** — Lightning Attention plus MLA as the main alternative family.

For each architecture, record only what matters for Small LLM:

- expected value at sub-1B scale;
- training and inference complexity;
- memory and KV-cache/state requirements;
- implementation difficulty;
- available kernels and T4 compatibility;
- evidence from similarly sized models or token budgets;
- whether the novelty teaches us something worth the added risk.

## 3. Select the model architecture

Compare at least these three candidates:

1. a modern dense GQA transformer baseline;
2. a 3:1 Gated DeltaNet/full-attention hybrid;
3. a 3:1 Gated DeltaNet-2/full-attention hybrid.

The decision should not be based only on which paper is newest. The selected model must be trainable on the actual T4, understandable enough to implement and debug, and still interesting enough to satisfy the learning goal of the project.

The likely direction is a dense hybrid decoder with three recurrent/linear-attention layers followed by one full-attention layer, repeated through the network. Keep the rest conservative: RMSNorm, pre-normalization, SwiGLU, tied embeddings, few or no biases, and no MoE.

Before freezing the choice, define a fallback: if the selected recurrent kernels are unstable or unacceptably slow on the T4, use the modern dense GQA transformer rather than blocking the whole project.

## 4. Start building the model

The implementation should be modular so that sequence mixers can be compared without rewriting the complete trainer.

Initial components:

- model configuration and geometry validation;
- GPT-2 token embeddings with a tied language-model head;
- RMSNorm;
- SwiGLU feed-forward network;
- a common sequence-mixer interface;
- a correct causal GQA attention baseline;
- the chosen recurrent or linear-attention mixer;
- decoder block and model stack;
- initialization rules;
- parameter-count reporting;
- basic forward, backward, causality, shape, and weight-tying tests.

Do not begin with a near-1B model. First construct a tiny smoke model that can overfit a tiny token sequence and verify that recurrent/chunked execution agrees with the reference path where applicable.

## Definition of done

Tomorrow is successful if:

- the final attention papers have been reduced to an implementation-relevant comparison;
- one primary architecture and one fallback have been selected and justified;
- the model geometry has been drafted;
- the initial model package exists with at least the dense baseline and test scaffolding started;
- no architecture decision depends on assumptions that still need to be measured on the T4.

---

# Architecture implementation TODO — July 31, 2026

The architecture investigation is complete enough to begin implementation. The following decisions and tasks supersede the exploratory items above where they conflict.

## Frozen implementation decisions

- [x] Keep PyTorch as the canonical framework for model code, autograd, training, checkpointing, and tests.
- [x] Keep a readable PyTorch GDN-2 recurrent implementation as the correctness oracle.
- [x] Use optimized Triton, CUDA, or library kernels only as replaceable backends behind PyTorch interfaces.
- [x] Use full MHA rather than GQA in the initial hybrid and baseline models.
- [x] Enable per-head QK-RMSNorm in MHA.
- [x] Enable an elementwise sigmoid attention-output gate before the output projection.
- [x] Use zero dropout in the initial model.
- [x] Use no additive embedding bias, no separate LM-head bias, and no ordinary linear biases outside faithful GDN-2 exceptions.
- [x] Use a 50,304-row internally padded tied embedding/output matrix.
- [x] Crop output logits to the semantic 50,257 classes before loss, evaluation probabilities, and sampling.

## Model package implementation

- [ ] Add `ModelConfig` with geometry and architecture validation.
- [ ] Add RMSNorm and RoPE modules with numerical tests.
- [ ] Add the bias-free SwiGLU FFN.
- [ ] Add gated full causal MHA with QK-RMSNorm.
- [ ] Add a readable PyTorch GDN-2 recurrence.
- [ ] Add short depthwise causal Q/K/V convolutions with kernel size 4.
- [ ] Add chunkwise/recurrent state and cache interfaces.
- [ ] Add an optimized-backend adapter without coupling model code to FLA internals.
- [ ] Add hybrid decoder blocks and the `[GDN-2, GDN-2, GDN-2, MHA]` stack builder.
- [ ] Add tied embeddings with aligned projection and semantic-logit cropping.
- [ ] Add exact parameter accounting by component.
- [ ] Add forward, backward, causality, shape, state-reset, weight-tying, and padded-vocabulary tests.
- [ ] Add chunkwise-versus-recurrent numerical-parity tests.
- [ ] Make the approximately 20M smoke model overfit a tiny deterministic sequence.

## Initialization decision experiment

- [ ] Compare a simple normal initialization against the reference GDN-2 Xavier-style initialization on a tiny model.
- [ ] Test depth-scaled residual output projections, approximately `1 / sqrt(2L)`.
- [ ] Preserve reference `A_log` and inverse-softplus `dt_bias` initialization in all candidates.
- [ ] Record forward activation variance, early loss, gradient norm, overflow events, and FP16 stability.
- [ ] Freeze the final initialization contract before the approximately 100M architecture trial.

## T4 GDN-2 kernel qualification

- [ ] Record `nvidia-smi --query-gpu=name,compute_cap`, CUDA, driver, PyTorch, and compiler versions on the target VPS.
- [ ] Run the PyTorch recurrent reference on the T4 and record correctness, memory, and throughput.
- [ ] Attempt installation and compilation of the current FLA GDN-2 backend.
- [ ] Determine whether any failure comes from Triton's compute-capability floor, unsupported operations/layouts, or kernel configuration.
- [ ] Benchmark forward and backward at the smoke and 100M geometries, context 2,048, microbatch 1.
- [ ] Benchmark recurrent single-token decoding and recurrent-state memory.
- [ ] Compare outputs and gradients against the PyTorch reference within explicit tolerances.
- [ ] Record FP16 overflow, NaN, and loss-scaling behavior; do not assume BF16 on T4.

## Possible T4 kernel side project

- [ ] If upstream kernels are unavailable or inadequate, isolate the recurrence hot loops and design a compute-capability-7.5 CUDA/CUTLASS backend.
- [ ] Keep the same PyTorch-facing API and autograd contract as the main model.
- [ ] Build correctness tests before optimization.
- [ ] Tune tile sizes, register pressure, shared memory, and recomputation for Turing's limits.
- [ ] Require a meaningful measured speedup over the readable PyTorch path before splitting it into a separate repository.
- [ ] Review the GDN-2 reference license carefully and implement from the published algorithm or permissive sources rather than copying restricted code.
- [ ] Consider publication only after reproducible T4 benchmarks, documentation, and tests exist.

## Main-project fallback

- [ ] Do not let the kernel side project block Small LLM.
- [ ] If no viable GDN-2 training backend exists on the T4, use the matched all-MHA fallback for trainer and dataset integration while continuing kernel work separately.
