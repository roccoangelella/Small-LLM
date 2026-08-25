# 100M/2B 10% capacity-aware S0 dataset wiring — 2026-08-25

## Status

ADR 0122's 10% S0 dataset recipe is wired on `main` for deterministic build and private Kaggle publication through the canonical `kaggle/launch_sft.py` surface.

The dataset has **not** been built or uploaded from this session because the project execution connector for `/home/ubuntu/Projects/Small-LLM` was unavailable, and no Kaggle connector was installed. Do not treat this evidence file as a publication-completion claim.

## Implemented build contract

The dedicated builder is:

```text
post_training/sft/s0_10pct_bundle.py
```

It is specific to the completed `100m-2b-data-001` parent and enforces:

```text
parent targets:       2,001,000,448
10% train ceiling:      200,100,044
planned instruction:    170,085,037
planned replay:          30,015,007
```

Planned instruction-source targets:

```text
smol-magpie-ultra-short  160,707,411
smol-contraints            4,026,530
smollm-rewrite-30k         3,762,301
smol-summarize-20k         1,588,795
```

The train builder uses those counts to derive capacity-aware instruction weights, reads each instruction stream without replacement, keeps ClimbMix at 15%, and fails if realized per-source targets drift by more than the configured packing tolerance.

Each build report is augmented with explicit realized source target shares in addition to source target counts.

## Frozen held-outs

Validation and test are rebuilt deterministically at the historical 4% requested held-out horizon:

```text
2,106,316 requested targets per split
```

They retain the old `75/10/7.5/7.5` instruction-source weights and seed 17. The build fails unless the resulting split manifests exactly match the completed 4% S0 identities:

```text
validation  26cb522729b4525498559d1ce131a181c30fd8fff573f3464e09030be803d09e
test        48e99ee51c201da398e227742ca7e023064a408c486cce16e20427d1ec7634d2
```

## Launcher routing

For the exact 100M/2B parent with `--sft-fraction 10%`, `kaggle/sft_scaled_runtime.py` routes bundle construction to the dedicated ADR-0122 builder. Other fractions retain the generic scaled builder.

The historical 4% profile keeps its existing launch pin. Only 10% bundle creation materializes the new implementation worktree, pinned to:

```text
fdfab079bacbb8a1098bdcee7451347cf28bc1f6
```

The resulting 10% experiment identities remain:

```text
SFT/W&B run:  100m-2b-sft-s0-10pct-001
Kaggle slug:  small-llm-100m-2b-sft-s0-10pct-001
```

## Private Kaggle publication

The existing `kaggle/sft_publish.py` path remains the publisher. It requires `KAGGLE_API_TOKEN`, stages and verifies the immutable bundle, uploads with KaggleHub, downloads the remote dataset back, checks byte-tree identity, verifies the bundle again, and fails if anonymous access succeeds.

Canonical build + private publish command:

```bash
python kaggle/launch_sft.py publish \
  --model 100M \
  --tokens 2B \
  --sft-fraction 10% \
  --replay-root /data/small-llm/20m-2b-ops/kaggle-dataset
```

With `KAGGLE_USERNAME` set, the default remote handle resolves to:

```text
$KAGGLE_USERNAME/small-llm-100m-2b-sft-s0-10pct-001
```

An explicit private handle can instead be passed through `--kaggle-dataset-handle owner/dataset`.

## Verification state

Focused regression coverage was added in `tests/test_sft_10pct_capacity_aware.py` for:

- isolated 10% run/dataset identities and exact 200,100,044-target budget;
- arithmetic of the capacity-aware instruction plan and 85/15 top-level mix;
- frozen validation/test target and manifest identities;
- exact 10% runtime detection and dedicated-builder routing;
- rejection of an existing 10% bundle whose recipe or frozen held-out identity drifts.

GitHub Actions did start for the implementation pushes, but the `unit-tests` job failed before any workflow steps were exposed (`steps=[]` and no downloadable job log). Therefore there is no successful CI execution claim for this wiring yet. The project-wide test workflow was already known to be unreliable/red for unrelated existing failures, but this particular no-step run also does not provide positive verification of the new tests.

## Implementation commits

```text
f68157e9  Wire capacity-aware 10 percent S0 bundle
c27e39fd  Route 10 percent S0 through capacity-aware builder
fdfab079  Test capacity-aware 10 percent S0 routing
dedfc464  Pin 10 percent bundle build implementation
```
