# Current Small-LLM Project Status

Last reviewed: 2026-09-03

## Repository and protocol state

- Repository: `roccoangelella/Small-LLM`.
- Current evaluation decision: ADR 0140 wires pretraining qualification v2 and SFT Behavior v2.
- Canonical sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
- Qualitative generation uses native per-case budgets.
- Pretraining EOS termination is not a metric.
- Teacher-forced confidence and masked SFT losses are diagnostics, not headline capability scores.

## Pretraining endpoints

| Model / data | Endpoint | Consumed target tokens | eval_core_v1 loss | Notes |
| --- | --- | ---: | ---: | --- |
| 20M / 100M | completed | ~100M | historical | early scaling point |
| 20M / 500M | completed | ~500M | historical | early scaling point |
| 20M / 2B | completed | ~2B | historical | same-size data scaling point |
| 100M / 2B | completed | 2,001,000,448 | 3.338815 | parent for canonical 100M/2B SFT |
| 100M / 10B | `step-00076294` | 10,000,007,168 | 3.129107 | run `100m-10b-deep-decay-from-step15500` |

The 100M/10B final intrinsic qualification reports perplexity 22.853570,
BPB 0.976699, top-1 0.418682, top-5 0.642991, and top-10 0.7165.

## SFT

### Canonical historical 100M/2B S0 trajectory

Canonical trajectory:
`100m-2b-sft-s0-10pct-peak3000-001`.

The 10% SFT run used approximately 200.1M target tokens. Relative to the
100M/2B parent it strongly improved masked SFT likelihood and generation
stopping/formatting, while slightly regressing `eval_core_v1` and achieving
only 1/30 strict passes in legacy behavior v1. This remains evidence that
teacher-forced SFT fit is not enough to establish instruction following.

### 100M/10B SFT

The 100M/10B SFT pipeline is wired and pinned to the current qualified
worktree/launch configuration. The same-data S0 recipe decision is recorded in
the project ADRs.

### Evaluation

SFT Behavior v2 is now the primary instruction-following evaluation design:

- 180 semantic tasks;
- six balanced families;
- L0 capability plus L1/L2/L3 progressively constrained variants;
- 720 total cases;
- 480 diagnostic cases;
- 240 held-out qualification cases;
- conditional compliance only where the same task's L0 semantic answer is correct;
- greedy primary plus sampled robustness over seeds 17/18/19;
- paired parent/SFT wins, losses, ties and exact McNemar statistics.

The legacy 30-case behavior suite remains only for longitudinal comparison.

## R-SFT

The production R-SFT path remains atomic-protocol based and retains its
reasoning-specific qualification. Base qualitative regressions now use native
prompt budgets; their general sampled view follows the project-wide
`temperature=1`, `top_p=1`, `top_k=0` contract. Reasoning pass@1 keeps its own
task-specific sampling protocol.

## Pretraining evaluation v2

Canonical full pretraining qualification now consists of:

1. frozen `eval_core_v1`;
2. six-task L20-Edu-style zero-shot conditional-likelihood evaluation using
   `lm-evaluation-harness==0.4.12`;
3. an expanded 120-prompt base-model suite with 100 mechanically scored cases
   and 20 readable qualitative continuations.

The six external tasks are ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI,
PIQA and WinoGrande. Full qualification uses all available benchmark examples.
Fast mode may limit the external tasks and is diagnostic only.

## Evaluation dependency boundary

`lm-evaluation-harness==0.4.12` is pinned in `requirements-eval.txt` rather
than the training lock. Evaluation tooling must not perturb the training
environment.
