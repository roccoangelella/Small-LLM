# Decisions and Ablations

_Last updated: 2026-07-31_

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
- Approximately 100M first substantive geometry: `d_model=512`, 20 layers, `d_ff=1408`, 8 MHA heads of dimension 64, and matching 8-head GDN key/value geometry.

## Rationale for newly frozen details

### PyTorch and optimized kernels

The project remains in PyTorch because the developer is already productive in it and because PyTorch supplies the model, autograd, optimizer, checkpoint, and testing surface needed by the project. Flash Linear Attention is itself a PyTorch package whose fast paths are implemented with Triton or other lower-level backends; adopting a kernel does not require switching model frameworks.

### GDN-2 short convolutions

The Q, K, and V short convolutions are depthwise causal 1D filters with initial kernel size 4. They add local token mixing before the recurrent state update. They are not a convolutional-network backbone and do not replace the decoder stack.

### Attention QK-Norm and output gating

These mechanisms are enabled in both the hybrid model and the all-MHA baseline. Keeping them matched prevents the architecture comparison from confounding recurrent versus softmax token mixing with unrelated attention-block changes.

### Vocabulary padding

The padded rows exist only to align the tied embedding/output matrix. They are not language-model classes. Cropping the aligned logits before cross-entropy and sampling retains hardware-friendly projection dimensions while defining the probability distribution over exactly the semantic vocabulary.

## Baseline

Maintain a modern all-MHA decoder baseline. Match as closely as possible:

- tokenizer and vocabulary;
- total parameters;
- depth or total compute, with differences documented;
- FFN type and width;
- normalization;
- QK-RMSNorm and attention output gating;
- RoPE;
- training tokens;
- batch and optimizer setup;
- data ordering and evaluation.

## Planned controlled ablations

These are not defaults and should change one important variable at a time:

- GDN-2:MHA ratios other than 3:1;
- GQA instead of MHA;
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
5. if the upstream path is unavailable or poor, evaluate a T4-compatible CUDA/CUTLASS implementation with the same PyTorch API;
6. consider publishing the T4-compatible GDN-2 kernel as a separate open-source side project only after correctness and measurable speedup are established.

The main Small LLM project must retain a working fallback and must not be blocked indefinitely by the side project.

## Still-open architecture details

- Exact global weight initialization.
- Exact gate initialization where the GDN-2 reference allows alternatives.
- Depth-dependent residual scaling.
- Exact larger-scale configurations beyond the frozen smoke and approximately 100M models.

## Decision standard

Do not add an architectural novelty merely because it appears in a large contemporary model. The hybrid recurrence is already the main experimental variable. New mechanisms require one of:

- a clear failure in the current design;
- a direct small-model result;
- a controlled T4 benchmark;
- a compelling implementation, memory, or serving constraint.

Scale decisions must follow measurements rather than parameter labels alone.
