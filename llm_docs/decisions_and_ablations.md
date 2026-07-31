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

## Frozen architecture defaults

- Dense decoder-only language model.
- Dominant Gated DeltaNet-2 mixer.
- Periodic ordinary MHA layers.
- 3:1 GDN-2-to-MHA pattern for the frozen initial models.
- Sequential pre-RMSNorm residual blocks.
- RMSNorm epsilon initially `1e-6`.
- Final RMSNorm before the tied LM head.
- Fixed full-head RoPE on Q and K in MHA only.
- No RoPE in initial GDN-2 layers.
- Dense SwiGLU FFN with SiLU gating in every block.
- Independent FFN weights in every layer.
- Tied input embeddings and output projection.
- Initial context 2,048.
- Approximately 20M smoke geometry.
- Approximately 100M first substantive geometry: `d_model=512`, 20 layers, `d_ff=1408`, 8 MHA heads of dimension 64, and matching 8-head GDN key/value geometry.

## Baseline

Maintain a modern all-MHA decoder baseline. Match as closely as possible:

- tokenizer and vocabulary;
- total parameters;
- depth or total compute, with differences documented;
- FFN type and width;
- normalization;
- RoPE;
- training tokens;
- batch and optimizer setup;
- data ordering and evaluation.

## Planned controlled ablations

These are not defaults and should change one important variable at a time:

- GDN-2:MHA ratios other than 3:1;
- GQA instead of MHA;
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

## Still-open architecture details

- Exact bias policy outside reference-required GDN-2 parameters.
- Dropout policy, though zero is the leading default.
- Exact weight and gate initialization.
- Depth-dependent residual scaling.
- QK-Norm or no QK-Norm.
- Attention output gating or ordinary output projection.
- Final internal vocabulary-padding implementation and invalid-logit masking details.
- Exact larger-scale configurations beyond the frozen smoke and approximately 100M models.

## Decision standard

Do not add an architectural novelty merely because it appears in a large contemporary model. The hybrid recurrence is already the main experimental variable. New mechanisms require one of:

- a clear failure in the current design;
- a direct small-model result;
- a controlled T4 benchmark;
- a compelling implementation, memory, or serving constraint.

Scale decisions must follow measurements rather than parameter labels alone.
