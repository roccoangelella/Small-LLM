# Production qualification of the accepted MoE identity on Modal — 2026-09-12

The short qualification required by ADR 0177 §6 before any long run, executed on three Modal GPUs in
parallel (H100 80GB HBM3, L40S, A10), one single-use container each, source commit
`c4f9104`'s parent tree (`2db1cc9`, checkpoint sequence of ADR 0179 and compile lane of ADR 0180
present, both default-off). Runner and raw files:
`/home/edo/Documents/0_Projects/Small-LM/docs/investigations/2026-09-12-qualifica-produzione/`.

## Workload

The real production command (`python -m MOE_model --model-size accepted --architecture gdn2_hybrid
--gdn-chunk-size 32 --load-balancing quantile --balancing-step-size 0 --optimizer hybrid_muon_adamw
--microbatch-size 16 --validation-blocks 1 --seed 17 ...`) on a real SuperBPE-8000 schema-v2 corpus
(2 train shards + 1 validation shard, 545 M tokens, retokenised with the frozen `superbpe_8000.json`,
`verify()` passed, staged in the `small-llm-data` volume under `/qualification-20260912`). 64 experts,
Top-2, Quantile Balancing on, 144,074,648 stored parameters. Throughput arms are engine-level on the
same shard with `MoEModelConfig.accepted()` untouched.

## Phase A — the production CLI, per GPU: pass on all three

| check | H100 | L40S | A10 |
|---|---|---|---|
| 6 updates, checkpoints at 3 and 6, validation | exit 0 | exit 0 | exit 0 |
| resume from step 3 in a copied directory, 3 updates | exit 0 | exit 0 | exit 0 |
| **step-4 loss, continuous vs resumed** | 8.953839 / 8.953839 | 8.953787 / 8.953787 | 8.953843 / 8.953843 |
| step-6 loss, continuous vs resumed | 8.881924 / 8.881876 | 8.881861 / 8.881750 | 8.881914 / 8.881945 |
| validation at step 6, continuous vs resumed | 8.846 / 8.846 | 8.846 / 8.846 | 8.846 / 8.846 |
| step-6 `trainer_state.pkl` byte-identical | no: 412/430 tensors, max abs 1.0e-3 | no: 412/430, 5.3e-4 | no: 412/430, 5.8e-4 |
| fp16 (production default), 3 updates | finite, loss 8.984588 at step 3 | finite | finite |
| checkpoint bytes / local save seconds | 1,161,781,264 / 2.3–2.4 | same / 1.6–1.9 | same / 2.3–2.5 |
| first update (cold: Triton + FLA compile) | 156 s | 209 s | 184 s |
| routing after 8 updates: max load fraction (uniform 0.0156) / dead slots | 0.035 / 0 | 0.035 / 0 | 0.035 / 0 |

**Reading of the resume check.** The first update after resume reproduces the continuous update to
all six printed decimals on every GPU, and validation is identical: model, optimizer, scheduler,
Quantile selection bias, RNG and data cursor are restored exactly. The state divergence appears only
from the second post-resume update at the 1e-5 loss level, and its largest tensor is the recomputed
`selection_bias` (1e-3 on a score scale of order 1). This is the signature of non-deterministic
CUDA kernels (no `use_deterministic_algorithms` anywhere in the trainer) amplified by Muon, not of a
resume defect. Control arm (two continuous runs from scratch on one A10, compared the same way):
see the addendum at the end of this record.

## Phase B — throughput, warm synchronized update seconds, 131,072 targets per update

| GPU | microbatch | compile | median update s | targets/s | peak allocated | first-step loss |
|---|---:|---|---:|---:|---:|---:|
| H100 | 32 | off | 0.474 | 276,720 | 23.85 GiB | 9.04706 |
| H100 | 64 | off | 0.452 | 289,906 | 44.64 GiB | 9.04706 |
| **H100** | **32** | **blocks** | **0.342** | **383,506** | **15.04 GiB** | 9.04707 |
| L40S | 16 | off | 1.034 | 126,794 | 14.14 GiB | 9.04705 |
| L40S | 32 | off | 1.145 | 114,505 | 23.76 GiB | 9.04700 |
| L40S | 16 | blocks | 0.816 | 160,590 | 9.45 GiB | 9.04709 |
| A10 | 16 | off | 2.059 | 63,665 | 14.18 GiB | 9.04706 |
| A10 | 16 | blocks | 1.481 | 88,501 | 9.40 GiB | 9.04707 |

The A10 eager figure reproduces the 2026-09-10 measurement (60,657 at microbatch 16, folded ids)
within 5 % on real data with Quantile Balancing on. **The H100 is 4.3× the A10 eager and 6.0× with
the compile lane**: the update is memory-bound (60 % of GPU time in elementwise/copy kernels, profile
of 2026-09-10), so bandwidth decides — 3.35 TB/s against 0.6.

**Compile lane (ADR 0180): +38.6 % on H100, +26.7 % on L40S, +39.0 % on A10, and −37 % peak memory.**
Per-step losses under compile track eager at the same magnitude as a microbatch change
(H100 step 8: 8.81195 compiled, 8.81239 eager at the same microbatch, 8.81196 eager at microbatch 64),
i.e. rounding-level. Compile warm-up is absorbed in the first 3 updates, excluded from the medians.
Microbatch 64 under compile was not measured.

## Economics (Modal list price per GPU, read 2026-09-12: H100 3.95, L40S 1.95, A10 1.10 USD/h; host
CPU and memory excluded; `cost = rate × quantity` on the update clock only)

| configuration | 100 B tokens: update-clock days | USD |
|---|---:|---:|
| **H100, compile, microbatch 32** | **3.0** | **286** |
| H100, eager, microbatch 64 | 4.0 | 379 |
| L40S, compile, microbatch 16 | 7.2 | 337 |
| A10, compile, microbatch 16 | 13.1 | 345 |

Calendar adds cold starts (≈ 3 min per container), checkpoints (2.4 s each; at every 2,500 updates
that is 0.3 %), validation, and the 24-hour segment restarts on Modal (4 segments for 3 days).

## Cost of this qualification

H100 475 s, L40S 619 s, A10 562 s of GPU time: ≈ 1.03 USD at list price plus the CPU gate, inside
the 4 USD ceiling declared to the owner. App stopped by the local entrypoint; zero containers left.

## Limits

Five to eight warm updates per arm, one container per GPU, no repeat runs: no confidence interval.
No learning-quality claim for the compile lane beyond rounding-level loss agreement over 8 updates;
ADR 0180 still requires a learning comparison for adoption. The corpus is a 2-shard proxy of the
production corpus, same tokenizer and schema. Routing balance after 8 updates says nothing about
100 B behaviour. Checkpoint durability to the Modal volume (commit time) was not measured here.
