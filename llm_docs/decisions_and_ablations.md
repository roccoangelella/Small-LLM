# Decisions and Ablations

_Last updated: 2026-08-01_

## Documentation decision

The topic files under `llm_docs/` are the sole source of truth for project decisions, technical contracts, current status, and open questions. The former monolithic `LLM_PROJECT_MEMORY.md` file has been retired after its information was consolidated into the topic documents.

When replacing a decision, record the old default, the new default, the reason, and the evidence. Do not silently erase superseded reasoning.

## Frozen project and data defaults

- Train a decoder-only language model below 1B parameters from random initialization.
- Use a geometry-scalable model implementation rather than a single hard-coded final size.
- Use the pinned `nvidia/Nemotron-ClimbMix` revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`.
- Accept clusters 1–10 and 12–20; exclude cluster 11.
- Use the GPT-2 byte-level BPE IDs already present in the corpus.
- Use exact empirical source-token mixture weights conditioned on cluster 11 being excluded.
- Keep mixture accounting continuous across documents, batches, shards, checkpoints, interruptions, and resumes.
- Use context+1 packing with stride equal to the context length.
- Use personal Google Drive as the durable dataset mirror, not as the random-access training filesystem.
- Overlap first-pass dataset preparation and model training after the operational gates pass.

## Frozen implementation defaults

- PyTorch is the canonical model and training framework.
- Keep a readable PyTorch GDN-2 recurrence as the correctness oracle.
- Treat Triton, CUDA, and external library kernels as replaceable optimized backends behind PyTorch interfaces.
- Do not let an optimized kernel redefine the mathematical contract.
- Require chunkwise and recurrent optimized paths to agree numerically with the reference path.
- Treat one-shot, segmented, and tokenwise execution of the current Python recurrent loop as recurrence/cache tests, not as evidence that a parallel chunkwise training path exists.
- Require a real chunkwise or otherwise parallel training backend, plus output and gradient parity against the reference recurrence, before accepting GDN-2 for substantive training.

## Frozen architecture defaults

- Dense decoder-only language model.
- Dominant Gated DeltaNet-2 mixer.
- Periodic full MHA layers.
- 3:1 GDN-2-to-MHA pattern for the frozen initial models.
- Sequential pre-RMSNorm residual blocks.
- RMSNorm epsilon initially `1e-6`.
- Final RMSNorm before the tied LM head.
- Fixed full-head RoPE on Q and K in MHA only.
- No RoPE in initial GDN-2 layers.
- Per-head QK-RMSNorm in full-attention layers.
- Elementwise sigmoid output gating in full-attention layers.
- Dense SwiGLU FFN with SiLU gating in every block.
- Independent FFN weights in every layer.
- Zero dropout throughout the initial model.
- Bias-free ordinary embedding, MHA, FFN, and projection paths.
- Retain only reference-required GDN-2 biases and bias-like offsets.
- Tied input embeddings and output projection.
- Semantic vocabulary 50,257 with an internally padded 50,304-row tied matrix.
- Crop aligned output logits to the first 50,257 classes before loss, evaluation probabilities, and sampling.
- Initial context 2,048.
- Approximately 20M smoke geometry.
- Approximately 100M first substantive geometry: `d_model=512`, 20 layers, hybrid `d_ff=1408`, 8 MHA heads of dimension 64, and matching 8-head GDN key/value geometry.
- Parameter-matched Plan B and Plan C transformer schedules use the same derived FFN width; at the substantive geometry both use `d_ff=1603`.

## Frozen fallback hierarchy

The primary architecture remains:

```text
[GDN-2, GDN-2, GDN-2, gated full MHA] × N
```

Fallbacks are ordered as follows.

### Plan A.5: Gated DeltaNet v1 hybrid

Use the same 3:1 pattern with ordinary Gated DeltaNet only when GDN-2 is the specific problem and a stable, efficient GDN-v1 training kernel works on the T4:

```text
[GDN, GDN, GDN, gated full MHA] × N
```

This is the closest architectural substitute, but it is not considered operationally safe until its kernel is qualified on the actual T4.

### Plan B: sliding-window/global gated-attention transformer

If recurrent or linear-attention training kernels are unavailable or inadequate, use:

```text
[SWA-512, SWA-512, SWA-512, gated full MHA] × N
```

`SWA-512` is causal sliding-window attention over the current token and at most the previous 511 tokens. It uses the same MHA projections, per-head QK-RMSNorm, RoPE, elementwise sigmoid output gate, normalization, bias policy, and parameter geometry as a full-attention block; only the attention mask differs. Every fourth layer remains full causal attention, preserving periodic unrestricted retrieval.

Plan B is the preferred operational fallback because it retains a local/global hybrid structure, requires no recurrent state or specialized linear-attention kernel, and is straightforward to implement with standard PyTorch attention primitives. For controlled comparison it uses the same parameter-matching FFN rule as Plan C.

### Plan C: all-gated-MHA transformer

The final and simplest fallback is a parameter-matched stack of gated full causal MHA layers:

```text
[gated full MHA] × L
```

Plan C is also the mandatory scientific baseline. It remains available even when the primary hybrid works, because architecture claims require comparison against a matched modern transformer.

The previous ordering placed all-MHA before the sliding-window/global transformer. On 2026-07-31 the user explicitly inverted those priorities: the local/global sliding-window transformer is now Plan B, and all-MHA is Plan C while retaining its baseline role.

## Rationale for newly frozen details

### Plan A.5 implementation boundary

The configuration can name the Plan A.5 schedule, but the model builder refuses to instantiate it until a real GDN-v1 implementation is supplied. GDN-v1 and GDN-2 are different recurrence algorithms. Reusing the existing GDN-2 module while labeling the run “GDN-v1” would produce a runnable model whose name, experimental condition, and actual mathematics disagree. Failing loudly prevents that silent substitution. Plan A.5 becomes runnable only after a separately implemented and T4-qualified GDN-v1 backend exists.

### Transformer fallback FFN matching

The old implementation widened only Plan C to the closest total-parameter match and left Plan B at the hybrid's `d_ff`. On 2026-08-01 the user replaced that asymmetry: Plan B and Plan C now use the same derived transformer FFN width.

The reason is structural. Both schedules replace the same number of GDN-2 mixers with gated attention mixers, and SWA-512 differs from full MHA only by its mask. Their learned mixer parameter counts are therefore identical, so there is no architecture-based reason for their FFN widths to differ. At the substantive geometry both use `d_ff=1603`, yielding 101,237,760 parameters versus 101,252,280 for the primary hybrid, a 14,520-parameter difference (about 0.014%). The hybrid remains at its frozen `d_ff=1408`.

### Recurrent reference versus chunkwise training

The current PyTorch GDN-2 oracle updates its matrix state sequentially for each token. Calling it once on a 2,048-token tensor still executes that serial recurrence inside Python. Splitting the same sequence into segments or single tokens and obtaining the same result proves causality, state reset, convolution-history handling, and cache correctness.

It does not prove the existence of the training path described in the architecture design. A production chunkwise path reorganizes the same recurrence into GPU-friendly chunks or scans so many token operations can run concurrently while preserving exact causal state transitions and supporting efficient backward propagation. That path needs its own implementation or qualified external kernel and must be compared against the oracle for outputs, final states, and gradients. Until then, the mathematics has a readable correctness reference, but GDN-2 training throughput at context 2,048 is not qualified.

### Optional SWA-512 implementation surface

The implementation work includes a 512-token sliding-window attention option because it is useful to exercise the shared attention interface. This does **not** replace the frozen initial full-causal MHA contract: smoke and substantive hybrid configurations keep unrestricted causal attention. SWA-512 is opt-in only, has no bearing on the primary hybrid, and is the defined Plan B fallback.

### PyTorch and optimized kernels

The project remains in PyTorch because the developer is already productive in it and because PyTorch supplies the model, autograd, optimizer, checkpoint, and testing surface needed by the project. Flash Linear Attention is itself a PyTorch package whose fast paths are implemented with Triton or other lower-level backends; adopting a kernel does not require switching model frameworks.

### GDN-2 short convolutions

The Q, K, and V short convolutions are depthwise causal 1D filters with initial kernel size 4. They add local token mixing before the recurrent state update. They are not a convolutional-network backbone and do not replace the decoder stack.

### Attention QK-Norm and output gating

These mechanisms are enabled in the hybrid model and every transformer fallback or baseline. Keeping them matched prevents architecture comparisons from confounding the sequence mixer with unrelated attention-block changes.

### Vocabulary padding

The padded rows exist only to align the tied embedding/output matrix. They are not language-model classes. Cropping the aligned logits before cross-entropy and sampling retains hardware-friendly projection dimensions while defining the probability distribution over exactly the semantic vocabulary.

## Baselines and comparison contract

Maintain both transformer references when resources permit:

1. Plan B local/global sliding-window transformer, for an efficient kernel-independent hybrid comparison;
2. Plan C all-MHA decoder, as the clean scientific baseline.

Match as closely as possible:

- tokenizer and vocabulary;
- total parameters;
- depth or total compute, with differences documented;
- FFN type and width between Plan B and Plan C;
- normalization;
- QK-RMSNorm and attention output gating;
- RoPE;
- training tokens;
- batch and optimizer setup;
- data ordering and evaluation.

The initial implementation replaces the hybrid's GDN-2 mixers with attention and widens the transformer schedules' SwiGLU branches to the closest integral total-parameter match. At the substantive geometry Plan B and Plan C both use `d_ff=1603`, leaving 14,520 parameters (about 0.014%) of unavoidable integer-width difference. The frozen primary hybrid uses `d_ff=1408` unchanged.

## Planned controlled ablations

These are not defaults and should change one important variable at a time:

- GDN-2:MHA ratios other than 3:1;
- GQA instead of MHA;
- sliding-window sizes other than 512;
- different SWA-to-full-attention ratios;
- QK-Norm disabled;
- attention output gate disabled or changed;
- partial RoPE or NoPE in MHA;
- RoPE inside GDN-2;
- learned rotary frequencies;
- negative-eigenvalue GDN-2 mode;
- grouped or expanded GDN value heads;
- GDN short convolution disabled or resized;
- no final RMSNorm;
- different FFN expansion ratios;
- an aligned transformer FFN width such as 1,600 instead of the exact-match width 1,603;
- expanded MHA Q/K/V projection width;
- longer contexts;
- nonzero dropout;
- bias variants;
- residual scaling and initialization variants.

## T4 kernel qualification and possible side project

The target T4 is a Turing, compute-capability-7.5 GPU. Current upstream Triton documents NVIDIA support beginning at compute capability 8.0, while the current Flash Linear Attention CUDA extra depends on Triton. Therefore T4 support must be treated as an empirical and possibly unsupported path rather than assumed.

The required sequence is:

1. run the PyTorch reference recurrence on the T4;
2. attempt the current FLA GDN-2 backend and record installation, compilation, correctness, and performance results;
3. inspect whether failure is a hard Triton architecture restriction, an unsupported instruction/layout choice, or only an autotuning/configuration issue;
4. benchmark forward, backward, chunkwise training, recurrent decoding, peak memory, and FP16 stability;
5. compare chunkwise outputs, final states, and gradients against the PyTorch recurrent oracle within explicit tolerances;
6. if the upstream path is unavailable or poor, evaluate a T4-compatible CUDA/CUTLASS implementation with the same PyTorch API;
7. consider publishing the T4-compatible GDN-2 kernel as a separate open-source side project only after correctness and measurable speedup are established.

The main Small LLM project must not be blocked indefinitely by the kernel side project. Plan B requires no recurrent kernel and is the preferred operational fallback; Plan C remains the simplest last resort and scientific reference.

## Still-open architecture details

- Exact global weight initialization.
- Exact gate initialization where the GDN-2 reference allows alternatives.
- Depth-dependent residual scaling.
- Exact larger-scale configurations beyond the frozen smoke and approximately 100M models.

These details do not block implementation of the model package or approximately 20M smoke configuration. Initialization is deliberately resolved by a tiny controlled implementation test rather than additional paper study.

## Decision standard

Do not add an architectural novelty merely because it appears in a large contemporary model. The hybrid recurrence is already the main experimental variable. New mechanisms require one of:

- a clear failure in the current design;
- a direct small-model result;
- a controlled T4 benchmark;
- a compelling implementation, memory, or serving constraint.

Scale decisions must follow measurements rather than parameter labels alone.
