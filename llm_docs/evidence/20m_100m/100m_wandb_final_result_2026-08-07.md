# 20M / 100M W&B Final Training Result — 2026-08-07

## Evidence identity

This report records the completed W&B training export supplied after the 20M-model / approximately-100M-token run.

```text
W&B run ID: 20m-100m-data-004
W&B run name: 20M model on 100M tokens
W&B state: finished
export archive SHA-256: 753a852c4e19894c6f7267a3f977671c03be602ab1e3f8a28bba08760e3fb5ef
first logged timestamp: 2026-08-06 08:33:23 UTC
final logged timestamp: 2026-08-07 10:55:33 UTC
wall-clock span represented by W&B: 26.37 h, including restarts and downtime
```

The final-session W&B configuration binds:

```text
model: 20,637,592-parameter gdn2_hybrid
context: 2,048
seed: 17
precision: FP16
training microbatch: 4
sequences per optimizer block: 16
optimizer: hybrid Muon + AdamW
peak LR: 3e-4
minimum LR ratio: 0.1
weight decay: 0.1
configured max GDN chunk: 32
validation cadence: 250 successful updates
local checkpoint cadence: 250 successful updates
verified remote publication cadence: 250 successful updates
final recovery worktree: 8e3cd9cb149facc5fa28e8108a70304c1f8c1c15
```

Dataset identities in the final W&B configuration:

```text
manifest SHA-256: f8e3bb926a793d171cff44f846c48ccc8acb749d6e0eaf2fb0bbcea8358c9077
drive manifest SHA-256: d6d958035934eaf9e6b3d07553e331a92fc9849e2794f5e9d2bc9e6c73e34ef6
```

## Completion

The finite schedule completed successfully at optimizer update 3,053 / block 3,052.

```text
final consumed training target tokens: 100,018,176
final block target tokens: 10,240
final block sequences: 5
final LR: 3.0e-5
final training loss: 4.423979
final held-out validation loss: 4.252758495143203
final held-out validation perplexity: 70.29906475797992
final validation target tokens: 124,928 across 4 blocks
final checkpoint: step-00003053
final checkpoint local size: 216,853,837 bytes
final checkpoint remote publication: verified, final=true
final remote publication elapsed: 13.863 s
```

The last training block is partial and unusually small, so its single loss `4.423979` is not a useful estimate of the terminal training distribution. The mean loss over the final approximately-100 logged updates is about `4.098`.

## Exact WSD schedule observed from LR telemetry

The logged learning rate reaches the configured peak at update 153, remains exactly at `3e-4` through update 2,442, and starts decaying at update 2,443. It ends at the configured 0.1 ratio, `3e-5`.

```text
warmup: updates 1–153
stable peak-LR region: updates 154–2,442
decay: updates 2,443–3,053
```

## Held-out validation trajectory

Every recorded validation point improved over the preceding point. There is no validation reversal or plateau before the finite schedule ends.

| Update | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 250 | 6.517897 | 677.153 |
| 500 | 5.762809 | 318.241 |
| 750 | 5.356173 | 211.912 |
| 1,000 | 5.113110 | 166.186 |
| 1,250 | 4.884521 | 132.227 |
| 1,500 | 4.725339 | 112.769 |
| 1,750 | 4.632458 | 102.766 |
| 2,000 | 4.557162 | 95.313 |
| 2,250 | 4.450810 | 85.696 |
| 2,500 | 4.395521 | 81.087 |
| 2,750 | 4.324534 | 75.530 |
| 3,000 | 4.260713 | 70.861 |
| 3,053 | 4.252758 | 70.299 |

From update 250 to the final checkpoint, validation loss fell by `2.26514` and perplexity fell by a factor of approximately `9.63x`.

The important scaling observation is the tail: validation loss still fell from `4.450810` at update 2,250 to `4.252758` at the end, a further `0.19805` improvement. Even after the decay region was well underway, it fell from `4.395521` at update 2,500 to `4.252758`, another `0.14276`. The 100M run therefore does **not** provide evidence that this 20M model had saturated its data benefit at 100M tokens. Marginal gains were diminishing, but still measurable and monotonic at the end. This supports the already-approved fresh 500M characterization run.

The historical 10M anchor ended at validation loss `6.136690` / perplexity `462.520157`. That is useful directional context, but it is not treated here as a strict same-validation-corpus comparison because the finite 10M and 100M dataset artifacts do not expose identical held-out block sets.

## FP16 behavior

The completed trajectory records nine total FP16 overflow events. Successful steps that consumed overflow retries were:

```text
update 939: 1 retry, scale -> 32768
update 1066: 1 retry, scale -> 16384
update 1083: 1 retry, scale -> 8192
update 1199: 1 retry, scale -> 4096
update 1247: 1 retry, scale -> 2048
update 1498: 4 retries, scale -> 128
```

After update 1,498, no additional overflow event is recorded through update 3,053. The loss scale remains at `128` for the rest of the run. This is direct evidence that the adaptive overflow-retry repair did what it was intended to do: the formerly fatal block completed after four retries, and the resulting calibrated scale was stable for the remaining approximately half of training.

No successful logged update has a non-finite loss or gradient norm.

## Gradient behavior

Across the 3,041 unique successful updates for which primary train telemetry is present:

```text
median gradient norm: 0.9718
p95 gradient norm: 1.5728
p99 gradient norm: 2.5759
maximum gradient norm: 17.8926 at update 1,144
logged clipped updates: 1,236 / 3,041 = 40.6%
```

The largest spikes were clipped as designed. Clipping is common rather than exceptional, so later scaling work should continue to monitor the distribution rather than treating `max_grad_norm=1.0` as an inactive safeguard. The run nevertheless remains stable in loss and validation through completion.

## Memory and data supply

Primary logged training telemetry shows:

```text
maximum peak allocated VRAM: 8.996 GiB
maximum peak reserved VRAM: 9.127 GiB
median peak reserved VRAM: 8.832 GiB
median data-wait per logged update: 4.23 ms
p95 data-wait: 12.25 ms
maximum data-wait: 57.04 ms
```

The attached immutable Kaggle dataset was not starving the GPU. Data-wait time is negligible compared with update compute time.

## Major throughput finding

The run exposes a large compute-throughput regression as training progresses. On the accepted session path, representative median throughput is:

```text
updates 1–1,000:      ~3,830 target tok/s
updates 1,001–1,250:  ~3,366 target tok/s
updates 1,251–1,500:  ~3,289 target tok/s
updates 1,501–1,750:  ~3,116 target tok/s
updates 1,751–2,000:  ~2,084 target tok/s
updates 2,001–2,250:    ~988 target tok/s
updates 2,251–2,518*:   ~654 target tok/s
updates 2,519–2,750:    ~590 target tok/s
updates 2,751–3,000:    ~501 target tok/s
updates 3,001–3,053:    ~445 target tok/s
```

`*` The accepted W&B session has a telemetry gap inside this interval; see the evidence-integrity section below.

The late-run median is about `8.6x` slower than the first 1,000 updates. Held-out validation shows almost the same slowdown: validation elapsed time rises from approximately `27.55 s` at update 1,000 to `236.20 s` at the final checkpoint, about `8.57x`.

This slowdown is **not explained by data starvation** and it appears in both training and validation compute. It is strongly consistent with the correctness-first adaptive GDN-2 backend increasingly selecting smaller numerical subchunks as learned decay spans grow. The project reference explicitly states that adaptive subchunk frequency and throughput cost were previously unqualified. However, the W&B export does not log the actually selected GDN subchunk sizes or decay-span-trigger counts, so this report does **not** claim a proven causal attribution.

Operational consequence: before interpreting 500M wall-clock behavior, instrument selected GDN subchunk sizes / split counts / decay-span statistics. The 500M scientific experiment remains valid, but the current correctness-first backend has demonstrated a serious throughput risk that can dominate runtime if the same trend recurs or worsens.

## Resume/replay evidence

The single W&B run ID preserves multiple Kaggle sessions and replayed tails. The ordered history contains five global-step rollbacks, giving six training sessions. Restarts resume at updates 1,001, 1,251, 2,251, 2,519, and 3,001.

Across adjacent sessions there are exactly 630 successful global updates logged in both the discarded/replayed tail and the resumed run. For every one of those 630 overlaps:

```text
maximum absolute replay loss difference: 0
maximum absolute replay gradient-norm difference: 0
```

This is strong empirical evidence that exact restore/replay preserved the model/data/optimizer trajectory on the same T4/software path. Throughput is not identical across replays, as expected from runtime conditions, but the numerical training observables are.

## Evidence-integrity note

The W&B export contains 3,710 history rows because validation/checkpoint events and replayed session tails share the same run ID. It should not be treated as 3,710 distinct optimizer updates.

For learning-curve reconstruction, one successful primary row is available for 3,041 of the 3,053 unique global steps. Updates `2,473–2,484` are absent from the W&B export entirely. The accepted replay session also has a larger primary-telemetry gap from `2,263–2,484`; the earlier discarded session contains `2,263–2,472`, and all directly observed replay overlaps elsewhere are exactly deterministic, but those earlier throughput measurements are not substituted for accepted-session performance measurements.

No metric values are invented for the 12 globally missing updates.

## What this evidence does and does not establish

Established from this export:

- the 100M finite pretraining schedule completed;
- the final checkpoint was locally saved and verified remotely;
- final held-out validation is `4.252758 / 70.299 PPL`;
- validation was still improving at the end;
- the repaired FP16 calibration path stabilized at loss scale 128 with no later overflows;
- exact replay numerics are empirically supported across 630 duplicated successful updates;
- dataset wait is negligible;
- training/validation compute throughput degraded severely late in the run.

Not established by this export alone:

- free-generation quality;
- `eval_core_v1` fast/full results;
- teacher-forced rank/confidence diagnostics;
- the exact GDN adaptive subchunk frequency or a proven causal attribution for the throughput collapse.

Those remain separate final-evaluation / instrumentation tasks.
