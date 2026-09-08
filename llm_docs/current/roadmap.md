---
status: current
last_reviewed: 2026-09-08
---

# Current roadmap

## Current position

- The 20M scaling series through 2B is complete.
- 100M/2B pretraining is complete at 2,001,000,448 consumed target tokens.
- 100M/10B deep-decay pretraining is complete at `step-00076294` / 10,000,007,168 consumed target tokens.
- Evaluation v2 is active under ADRs 0140 and 0141.
- ADR 0160 selects approximately 200M parameters / 100B target tokens for the next major pretraining trajectory.
- ADR 0161 keeps the successful 100M/10B deep-decay/WSqD-style LR structure while modestly widening every major LR anchor.
- ADR 0162 freezes phase timing as a 10x proportional scaling of the successful 100M/10B trajectory: warmup endpoint step 38,147 (~5.000B targets), deep-decay start step 155,000 (20.31616B), aggressive-settle endpoint step 177,890 (23.31639808B), terminal-cooldown start step 732,420 (95.99975424B), and final step 762,940 (100.00007168B).
- ADR 0163 freezes the exact LR anchors at `3.5e-4 -> 8e-5 -> 8e-6 -> 4e-6`; with ADR 0162 timing, the calibrated long-decay exponent remains `~1.6270515945225403`.
- ADR 0164 restricts production wiring for this run to Modal H100 and Beam RTX4090; no Kaggle 200M/100B production launcher is authorized.
- ADR 0158 remains in force: the complete public-HF `100b-b64-dataset-001` corpus is built to 100B and the next major run is intended to consume the full corpus; paid providers do not build this corpus.

## Immediate priorities

1. Build and fully publish `100b-b64-dataset-001` to its dedicated public HF bucket, then verify its terminal manifest/READY/frontier and immutable shard inventory.
2. Preserve block-64/context-2048 global geometry, hybrid Muon+AdamW routing, FP16 qualified GDN-2 execution, exact dataset order, split latest/best checkpoint semantics, and cross-provider exact-resume identity unless explicitly superseded.
3. Run local tests plus Modal H100 and Beam RTX4090 import/data-stage/microbatch/training/resume smoke gates before any long dispatch.
4. Launch the long trajectory only after the completed-corpus and provider smoke gates pass.

## Wiring readiness findings

The accepted 200M/100B run is now wired without opening a Kaggle path:

- `100b-b64` resolves through one shared Modal/Beam scientific profile registry under `providers/`;
- the provider launchers enforce an explicit session-step budget and restrict 100B to Modal H100 / Beam RTX4090;
- `trainer --model-size expanded` constructs the frozen 200M geometry;
- trainer setup installs the accepted 200M/100B WSqD plan before checkpoint identity is computed;
- the generic trainer serializes exact-resume scheduler and optimizer state;
- compact `--profile 200M-100B` and legacy `--model 200M --tokens 100B` selectors resolve identically.

## Next operational gate

No further scientific decision is required before smoke qualification. The remaining gates are completed public-corpus verification and provider-specific import, staging, training, and exact-resume smoke results. Any proposed change to geometry, optimizer routing, LR anchors, dataset order, or provider boundary still requires an explicit superseding decision.

The accepted target implementation is:

- approximately 200M hybrid family geometry: `d_model=768`, `n_layers=20`, `d_ff=2048`, 12 x 64 attention/GDN heads, `[GDN-2, GDN-2, GDN-2, gated full MHA]` rhythm, context 2048;
- hybrid Muon + AdamW routing unchanged;
- block-64 global optimizer updates = 131,072 targets/update;
- exact 100B block-aligned endpoint = 100,000,071,680 targets / step 762,940;
- WSqD LR anchors `3.5e-4 -> 8e-5 -> 8e-6 -> 4e-6` at ADR 0162's proportional boundaries;
- long-decay `base_power ~= 1.6270515945225403`;
- Modal H100 and Beam RTX4090 only.

## Frozen boundaries still in force unless explicitly superseded

- Context remains 2,048 for the current comparison family.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- New dataset durability uses Hugging Face Storage Buckets, not Google Drive.
- Live exact-resume checkpoints and strict validation-loss best artifacts remain separate according to ADR 0132.
- Canonical sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
- Pretraining EOS termination is not a metric.
- Teacher-forced confidence and masked SFT losses remain diagnostics rather than headline capability scores.
