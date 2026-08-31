# 0133 — Record 100M/2B 10% SFT peak-through-3000 qualification result

Date: 2026-08-31
Status: Accepted

## Decision

Keep `100m-2b-sft-s0-10pct-peak3000-001` as the canonical 100M/2B 10% SFT checkpoint for follow-up chat and evaluation work, while recording that it is an SFT-adaptation win rather than a clean all-metric improvement over the pretrained parent.

Do not use the stale `100m-2b-sft-s0-10pct-longpeak-001` resolver identity, and do not prefer the older no-peak `100m-2b-sft-s0-10pct-001` trajectory for the 10% SFT profile.

## Evidence from full qualification JSON

Compared against pretrained parent `100m-2b-data-001` at `step-00015267` / `2,001,000,448` consumed tokens, the peak-through-3000 run `100m-2b-sft-s0-10pct-peak3000-001` at `step-00006219` / `200,099,738` consumed SFT tokens shows:

| Metric | Parent | Peak-through-3000 SFT | Delta, SFT - parent | Direction |
| --- | ---: | ---: | ---: | --- |
| eval_core_v1.loss | 3.3388147759 | 3.4202637300 | +0.0814489541 | worse retention |
| eval_core_v1.perplexity | 28.1857005300 | 30.5774781564 | +2.3917776264 | worse retention |
| eval_core_v1.bits_per_byte | 1.0421553917 | 1.0675783254 | +0.0254229337 | worse retention |
| top-1 accuracy | 0.3988752193 | 0.3951862589 | -0.0036889604 | slight drop |
| top-5 accuracy | 0.6181535102 | 0.6134755932 | -0.0046779171 | slight drop |
| top-10 accuracy | 0.6920412732 | 0.6872732160 | -0.0047680572 | slight drop |
| SFT validation loss | 2.3579566026 | 1.6417784840 | -0.7161781186 | strong SFT fit gain |
| SFT validation perplexity | 10.5693320255 | 5.1643460563 | -5.4049859692 | strong SFT fit gain |
| SFT test loss | 2.3487316845 | 1.6233234382 | -0.7254082463 | strong SFT fit gain |
| SFT test perplexity | 10.4722791447 | 5.0699118873 | -5.4023672574 | strong SFT fit gain |
| instruction pass rate | 0.0000000000 | 0.0333333333 | +0.0333333333 | one strict pass |
| instruction EOS termination | 0.0000000000 | 0.2333333333 | +0.2333333333 | better stopping |
| instruction runaway rate | 1.0000000000 | 0.7666666667 | -0.2333333333 | better stopping |
| instruction role leak rate | 0.1666666667 | 0.0000000000 | -0.1666666667 | better formatting safety |
| mean trigram repetition | 0.5323983170 | 0.0979393678 | -0.4344589492 | much less repetition |

Interpretation: peak-through-3000 is not free. It buys much better held-out SFT loss and observable instruction-behavior cleanup at the cost of a small but consistent frozen eval-core degradation relative to pretraining.

## Comparison with older 10% no-peak trajectory

The accessible W&B summary for the older no-peak `100m-2b-sft-s0-10pct-001` finished run reports:

| Metric | No-peak 10% | Peak-through-3000 10% | Delta, peak - no-peak | Direction |
| --- | ---: | ---: | ---: | --- |
| final training loss | 2.0145676136 | 1.8481694460 | -0.1663981676 | better fit |
| W&B validation loss | 1.7946823422 | 1.6006682965 | -0.1940140457 | better fit |
| W&B validation perplexity | 6.0175628923 | 4.9563436251 | -1.0612192672 | better fit |
| SFT behavior pass rate | 0.0333333333 | 0.0333333333 | 0.0000000000 | no strict-pass gain |
| SFT behavior EOS termination | 0.1333333333 | 0.2333333333 | +0.1000000000 | better stopping |
| SFT behavior runaway rate | 0.8666666667 | 0.7666666667 | -0.1000000000 | better stopping |

The older no-peak full evalcard was not located in the accessible repo/W&B/HF surfaces during this comparison, so eval-core retention against the older no-peak run is not asserted here. On the available shared SFT and behavior metrics, peak-through-3000 is strictly better or tied.

## Operational consequence

Use the peak-through-3000 checkpoint for the canonical 100M/2B 10% SFT route. When reporting it externally, call out the tradeoff explicitly: it is the better SFT-adapted model so far, but it slightly worsens base eval-core loss/perplexity and top-k accuracy versus the pretrained parent.
