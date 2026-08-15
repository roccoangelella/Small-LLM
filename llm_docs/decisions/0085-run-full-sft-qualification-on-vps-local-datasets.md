# ADR 0085: Run full SFT qualification on the VPS with local test datasets

- Status: Accepted
- Date: 2026-08-15

## Context and problem statement

The completed 100M / 2B SFT qualification had been launched through `kaggle/launch_sft.py eval`, which made the operational path depend on Kaggle dataset attachment/discovery even though the underlying evaluator is provider-neutral. The frozen `eval_core_v1` corpus and the SFT bundle are immutable test inputs and do not need Kaggle as a transport layer when a persistent VPS is available.

The repository `tests/` directory also mixed ordinary discovered tests, reusable fixtures, hardware qualification harnesses, and operational qualification data conventions without a clear separation of roles.

## Decision

The canonical 100M / 2B full parent-versus-SFT qualification moves to the VPS.

- Large qualification inputs live locally under `tests/test_datasets/` and are ignored by git.
- The expected local roots are `tests/test_datasets/eval_core_v1/` and `tests/test_datasets/100m-2b-sft-s0-001/`.
- `python -m tests.qualification.sft_100m_2b_vps --suite full` is the canonical launcher.
- The launcher verifies both local datasets before model evaluation, loads `.env` when present, and invokes the existing provider-neutral `post_training.sft.eval_suite` directly.
- Parent and SFT checkpoints remain verified Hugging Face artifacts by default; explicit local checkpoint directories remain supported.
- Kaggle dataset upload/attachment is no longer required for this qualification path.
- Ordinary `test_*.py` files remain at the `tests/` package root for current `unittest discover` compatibility. They are not mass-deleted based on age alone; removal requires evidence that the protected production path is gone or coverage is duplicated.
- Operational qualification launchers live under `tests/qualification/`, while local large corpora live under `tests/test_datasets/`.

## Consequences

- Re-running the full SFT qualification no longer requires publishing immutable test datasets to Kaggle.
- The persistent VPS can keep the frozen corpora once and reuse them across evaluations.
- Dataset corruption or incomplete copies fail before expensive model scoring because both bundle and eval-core verification run first.
- `.env` repository IDs and `HF_TOKEN` work directly with the VPS launcher without an external shell-export step.
- The tests directory has an explicit separation between discovered regressions, operational qualification launchers, and local data.
- Historical Kaggle/T4 paths may remain in the repository where they still protect or document historical runtime behavior, but they are not the canonical 100M / 2B SFT qualification route.
