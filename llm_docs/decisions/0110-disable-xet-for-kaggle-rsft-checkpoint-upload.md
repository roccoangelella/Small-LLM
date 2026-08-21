---
status: accepted
date: 2026-08-21
supersedes: null
---

# 0110 — Disable Xet for Kaggle R-SFT checkpoint upload

## Context and problem statement

The first complete 16,716-row R-SFT training pass reached optimizer step 417 and created the final local checkpoint, but rank zero was SIGKILLed during the final Hugging Face upload of the approximately 914 MB `trainer_state.pkl`. The Xet-backed upload stalled for roughly fifty minutes after reaching about 211 MB of new data. Rank one then failed at the control barrier because rank zero had disappeared.

The two-phase publication contract behaved correctly: small step-417 metadata files reached the repository, but the missing state blob prevented upload verification and `latest.json` remained on the fully verified step-250 checkpoint. The run is therefore resumable without accepting a partial final checkpoint.

## Decision outcome

For the fixed 100M/2B Kaggle R-SFT launcher, set `HF_HUB_DISABLE_XET=1` and `HF_HUB_DISABLE_PROGRESS_BARS=1` in the DDP process environment. This forces Hugging Face checkpoint uploads through the classic streaming HTTP/LFS path and avoids notebook progress-bar output during large uploads.

Keep the existing two-phase publication and exact checkpoint format unchanged. Do not move `latest.json` unless every uploaded file is present and SHA-verified.

## Consequences

### Positive

- Avoids the Xet transfer path implicated in the final step-417 stall/SIGKILL.
- Keeps large checkpoint upload memory/output behavior simpler in Kaggle batch sessions.
- Requires no change to model state, optimizer state, training geometry, or the pinned training implementation.
- A rerun of the same production run ID resumes from verified step 250 and replays only steps 251–417.

### Negative or limiting

- Classic LFS upload may transfer more bytes than Xet deduplication would.
- The already-computed first step-417 state cannot be reconstructed from the partial remote metadata alone; direct recovery requires the original Kaggle local `trainer_state.pkl` if that notebook output is still available.

## Validation

The R-SFT dry-run command must contain both upload environment variables. Focused Kaggle R-SFT tests must pass, and the pinned `huggingface_hub==1.5.0` runtime must report Xet disabled and progress bars disabled when those variables are set.
