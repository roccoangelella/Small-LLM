# Decisions and Ablations

_Last updated: 2026-07-31_

## Frozen architecture defaults

- Decoder-only dense language model.
- Geometry-scalable implementation.
- Dominant Gated DeltaNet-2 mixer.
- Periodic ordinary MHA layers.
- 3:1 GDN-2-to-MHA pattern.
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

- exact bias policy outside reference-required GDN-2 parameters;
- dropout policy, though zero is the leading default;
- exact weight and gate initialization;
- residual-branch scaling;
- QK-Norm or no QK-Norm;
- attention output gating or ordinary output projection;
- final internal vocabulary-padding implementation and invalid-logit masking details;
- exact larger-scale configurations beyond the frozen smoke and approximately 100M models.

## Decision standard

Do not add an architectural novelty merely because it appears in a large contemporary model. The hybrid recurrence is already the main experimental variable. New mechanisms require one of:

- a clear failure in the current design;
- a direct small-model result;
- a controlled T4 benchmark;
- a compelling implementation or serving constraint.
