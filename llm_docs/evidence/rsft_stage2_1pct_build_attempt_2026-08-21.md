# R-SFT Stage-2 1% build attempt — 2026-08-21

## Scope

ADR 0113 selects Superior Reasoning Stage 2 `instruction_following` as the additive source for the ADR-0112 nested 1% / 2% / 4% R-SFT scaling sweep. This evidence records the first wiring and build attempt. It deliberately distinguishes **pipeline readiness** from **a completed corpus artifact**.

## Memory gap corrected

Before ADR 0113, project memory documented why the first large production R-SFT used only Superior Reasoning Stage 1, but did not explicitly call out the availability of the dataset's Stage 2 as future scaling headroom. ADR 0113 now records Stage 2 as the selected expansion source while preserving the Stage-1 artifact and policy.

## Implemented Stage-2 lane

`post_training/R-SFT/dataset/scale_superior_reasoning.py` now provides a separate scaling path so the frozen Stage-1 reproduction code remains unchanged.

The Stage-2 producer reuses the Stage-1 production policy implementation directly and applies:

- config `stage2`, split `train`, followed by `domain == instruction_following` selection;
- strict `<think>...</think>` teacher-output parsing;
- normalized-prompt deduplication against the frozen 16,716-row base and within Stage 2;
- the unchanged primary-math / primary-code exclusion function;
- reserved reasoning-marker collision rejection;
- exact atomic 2,048-token serialized-context validation with no truncation;
- direct acceptance only for already-fitting examples;
- freezing of otherwise-clean over-context rows as adaptation candidates;
- loss-bearing assistant-target accounting using the three atomic marker tokens, reasoning text, answer text, and assistant EOS;
- exact projection of `build_atomic.py`'s deterministic 1% validation + 1% test partition per reasoning group before measuring train reasoning targets;
- append-only Stage-2 ordering by the deterministic instruction-stream `source_index`, making later 2%/4% source scans unable to insert newly discovered rows ahead of the frozen 1% prefix.

Focused tests live in `tests/test_reasoning_sft_stage2_scaling.py`. They cover Stage-2 train/domain selection, Stage-1 filter/context reuse, train-target budget semantics, and literal source-order nesting. The tests were committed, but this evidence does **not** claim a successful CI execution because the attempted GitHub Actions validation did not receive a visible run.

## Exact 1% target

The verified 100M/2B parent consumed 2,001,000,448 training targets.

```text
requested 1% total R-SFT targets:       20,010,004
requested reasoning train targets @90%: 18,009,004
existing expanded reasoning train:       12,077,733
additional reasoning train needed:         5,931,271
```

The retention lane remains 10% of the realized total and is selected later from the exact completed S0 instruction records. If the reasoning prefix lands exactly on 18,009,004 targets, the corresponding retention request is 2,001,000 targets and the total is 20,010,004.

## Real build attempts

Three execution routes were attempted from the available tooling:

1. **Hugging Face Jobs CPU** — the remote execution request was rejected before allocation with HTTP `402 Payment Required`. No Stage-2 scan ran there.
2. **GitHub Actions** — `.github/workflows/rsft-stage2-1pct.yml` was added and temporary PR #8 changed only a trigger marker. No PR workflow run became visible, including the repository's ordinary PR test workflow, so no runner executed the Stage-2 scan. This is not evidence of a pipeline/test failure. The temporary validation PR was not intended for merge.
3. **Direct Stage-2 object download into the local tool container** — the available large-file/Xet download path did not deliver the Stage-2 payload into the local runtime, so a local full-source scan could not be completed.

No Kaggle execution connector is available in the current session.

## Current result

**The Stage-2 scaling pipeline is wired on `main`, but the real 1% reasoning corpus has not yet been produced.** Do not report a row count, SHA-256, or completed 1% artifact until a networked project environment actually scans the canonical Stage-2 text source and freezes the resulting manifest.

The canonical first attempt in a networked repo checkout is:

```bash
python post_training/R-SFT/dataset/scale_superior_reasoning.py prepare-stage2 \
  --base-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --work-dir /path/to/rsft-stage2

python post_training/R-SFT/dataset/scale_superior_reasoning.py build-percent \
  --base-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --work-dir /path/to/rsft-stage2 \
  --output-jsonl /path/to/rsft-1pct/reasoning.jsonl \
  --percent 1
```

If context-fit Stage-2 instruction rows alone reach the 1% target, no Gemini calls are required. If they do not, the build must stop rather than relax policy: curate the frozen Stage-2 over-context candidates under the same keep/math/code/safety gate, run only the curated keepers through the ADR-0103/0106 Gemini-only Variant-D path, then rebuild with `--include-adapted`.
