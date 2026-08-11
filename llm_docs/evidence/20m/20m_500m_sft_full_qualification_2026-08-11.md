---
status: evidence
observed_at: 2026-08-11
run_id: 20m-500m-sft-s0-001
parent_run_id: 20m-500m-data-001
suite: full
---

# 20M / 500M-parent SFT full qualification

This record captures the completed full post-SFT qualification for the approximately-20M `gdn2_hybrid` model pretrained on approximately 500M loss-bearing tokens and then SFT-trained on the frozen S0 mixture.

## Identity

```text
parent checkpoint: step-00015264
parent consumed tokens: 500,156,416
SFT checkpoint: step-00000621
SFT consumed loss-bearing target tokens: 20,006,234
SFT bundle status: verified
SFT train targets: 20,006,234
SFT validation targets: 526,446
SFT test targets: 526,473
```

The requested 4%-of-parent ceiling was 20,006,256 targets, so the realized SFT trajectory finished 22 targets below that nominal ceiling because the immutable packed bundle contains 20,006,234 train loss-bearing targets.

## Held-out SFT objective

SFT substantially improved masked held-out SFT likelihood:

| metric | parent | SFT | delta (SFT-parent) |
|---|---:|---:|---:|
| validation loss | 3.212253 | 2.639931 | -0.572322 |
| validation perplexity | 24.834964 | 14.012232 | -10.822732 |
| test loss | 3.185668 | 2.609139 | -0.576529 |
| test perplexity | 24.183441 | 13.587347 | -10.596094 |

This is strong evidence that the SFT training objective itself was learned on the held-out SFT distribution.

## Base-capability retention (`eval_core_v1`)

The unchanged full base evaluation regressed modestly after SFT:

| metric | parent | SFT | delta (SFT-parent) |
|---|---:|---:|---:|
| loss | 4.007289 | 4.047304 | +0.040016 |
| perplexity | 54.997550 | 57.242943 | +2.245393 |
| bits/byte | 1.250808 | 1.263299 | +0.012490 |
| top-1 accuracy | 0.343950 | 0.339413 | -0.004536 |
| top-5 accuracy | 0.547491 | 0.542458 | -0.005033 |
| top-10 accuracy | 0.620226 | 0.615493 | -0.004734 |
| calibration ECE | 0.007731 | 0.020217 | +0.012486 |
| cluster-macro loss | 4.031874 | 4.070689 | +0.038815 |
| mixture-weighted cluster loss | 3.529477 | 3.589827 | +0.060350 |

Per-cluster loss worsened in 18 of 19 reported clusters; cluster 5 improved by approximately 0.0193 loss. All eight position buckets regressed. The base-retention cost is therefore broad rather than isolated to one position range or one cluster.

## Instruction behavior

The deterministic 30-case behavior suite did **not** show successful instruction acquisition:

```text
pass rate:                 0 / 30 = 0.0%
per-category pass rates:   0.0% in every category
empty rate:                0.0%
EOS termination rate:      0.0%
runaway rate:              100.0%
role leak rate:            0.0%
mean response tokens:      48.53
mean trigram repetition:   0.4626
```

The parent also scored 0/30 with 100% runaway. SFT reduced mean trigram repetition from 0.5742 to 0.4626 (delta -0.1116) and changed response style, but it did not convert that change into any passing deterministic instruction-behavior cases or EOS termination.

Representative failures remain fundamental rather than marginal: factual QA is often confidently wrong, arithmetic is not executed, exact transformations/extractions/format constraints fail, multi-turn corrections fail, reasoning tasks fail, and the safe-refusal case does not refuse.

## Interpretation

The 500M-parent S0 experiment demonstrates a sharp separation between **SFT-distribution likelihood learning** and **usable instruction-following behavior** at this model/data scale. The held-out masked SFT loss improves strongly while the deterministic behavior score stays at 0%, runaway remains 100%, EOS termination remains 0%, and unchanged base language modeling degrades modestly.

Therefore this run should be treated as a **failed behavioral SFT qualification**, not as evidence that the current S0 recipe is ready to be promoted unchanged to larger parents solely because its validation/test SFT loss is lower. The result does, however, validate that the SFT pipeline can optimize the intended masked target distribution and supplies a concrete baseline for the next SFT design decision.

No single master score was used; interpretation follows the frozen policy of inspecting instruction acquisition and base-capability retention together.

## Source artifact

Original qualification artifact supplied by the user: `post-sft-full-qualification.json`.

Report SHA256:

```text
447e3f5eb7f22d6d3270a598557cb3456a1c5911c555d8fddac906a7309d9139
```
