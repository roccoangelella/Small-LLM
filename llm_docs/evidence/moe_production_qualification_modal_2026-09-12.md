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
| validation at step 6, continuous vs resumed (differs at 8e-5) | 8.846 / 8.846 | 8.846 / 8.846 | 8.846 / 8.846 |
| step-6 `trainer_state.pkl` byte-identical | no: 412/430 tensors, max abs 1.0e-3 | no: 412/430, 5.3e-4 | no: 412/430, 5.8e-4 |
| fp16 (production default), 3 updates | finite, loss 8.984588 at step 3 | finite | finite |
| checkpoint bytes / local save seconds | 1,161,781,264 / 2.3–2.4 | same / 1.6–1.9 | same / 2.3–2.5 |
| first update (cold: Triton + FLA compile) | 156 s | 209 s | 184 s |
| routing after 8 updates: max load fraction (uniform 0.0156) / dead slots | 0.035 / 0 | 0.035 / 0 | 0.035 / 0 |

**Reading of the resume check.** The first update after resume reproduces the continuous update to
all six printed decimals on every GPU, and validation is identical: model, optimizer, scheduler,
Quantile selection bias, RNG and data cursor were restored with no defect detectable within the CUDA noise measured by the control below. The state divergence appears only
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

## Addendum — nondeterminism control, one A10, same day

Two continuous 6-update runs from scratch, identical command, seed and data, in the same container:
step-6 `trainer_state.pkl` differs on the same 412 of 430 tensors with max abs difference **2.4e-3**,
larger than the 3.5e-4 between the continuous run and its resume in that container (and than the
5e-4–1e-3 seen on the three GPUs above). Run-to-run kernel nondeterminism therefore fully accounts for
the post-resume divergence; checkpoint/resume is exact to the limit the hardware allows. The 18
identical tensors are the integer routing counters and step bookkeeping. Cost ≈ 0.12 USD.

## Addendum — H100 compile lane at microbatch 64 and under CUDA graphs, same day

| microbatch | compile | median update s | targets/s | peak allocated |
|---:|---|---:|---:|---:|
| 64 | off | 0.462 | 283,892 | 44.64 GiB |
| **64** | **blocks** | **0.330** | **397,698** | 26.92 GiB |
| 32 | blocks + `mode="reduce-overhead"` | failed | — | — |

Microbatch 64 under the compile lane adds +3.7 % over microbatch 32 (383,506): the update is no
longer microbatch-limited. CUDA graphs fail on the second microbatch with *"accessing tensor
output of CUDAGraphs that has been overwritten by a subsequent run"* at the padded expert GEMM
(`MOE_model/model.py:160`): the compiled block's outputs are consumed across microbatches without
`torch.compiler.cudagraph_mark_step_begin()` and without cloning. Fixable, but with 7–18 graph
breaks per block the expected gain is small; parked. Best measured production configuration:
**H100, `--compile blocks`, microbatch 64, BF16: 100 B tokens in 2.9 update-clock days, ≈ 276 USD
of GPU at list price.** Cost of this follow-up ≈ 0.39 USD.

## Addendum — after external review (Astra, same day)

- **Common update window.** The main table's medians used steps 2–8 for eager and 4–12 for the
  compile lane. Recomputed on steps 4–8 for both arms: H100 278,873 → 381,138 (+36.7 %), L40S
  126,931 → 160,243 (+26.2 %), A10 63,882 → 88,476 (+38.5 %). Conclusions unchanged.
- **What the resume check does and does not prove.** The identical step-4 loss proves the model
  weights, selection bias, RNG and data cursor were restored; it is computed before the first
  post-resume optimizer update, so it says nothing about the optimizer moments. The first
  post-resume update is step 5. On the A10 control, the continuous-vs-resumed step-5 loss differs
  by 4.3e-5 while two continuous runs from scratch differ by 7.5e-5 at the same step: the first
  resumed update lies inside run-to-run nondeterminism, which is the evidence that the optimizer
  state was restored. Step-4 gradient norms differ at 1e-6 for the same reason (H100 0.4670625925
  vs 0.4670630991). Resume **with the compile lane armed** has not been exercised: pending.
- **Compile lane, open before adoption** (ADR 0180): the compiled backward's autocast contract
  (torch 2.10 defaults `backward_pass_autocast="same_as_forward"`, the eager step calls
  `.backward()` outside autocast) is being pinned to the eager behaviour; the selection bias is
  bit-identical to eager only on identical inputs (CPU), on H100 it differs at rounding level
  (0.0332623720 vs 0.0332735777 after the first update); a per-parameter gradient comparison on
  GPU and a compiled resume remain to be run.

## Addendum — Beam RTX 4090, same day (source `db9c4f1`, runner `beam/moe_qualification.py`)

Same protocol on Beam (serverless function, `LEGACY_SERVERLESS_IMAGE`, torch 2.10.0+cu126, corpus
staged in the Beam `small-llm-data` volume). Production CLI: continuous exit 0, resume exit 0,
step-4 loss identical (8.953808 / 8.953808), step-5 within nondeterminism (3.0e-5), state differs on
412/430 tensors with max 6.9e-4; fp16 finite; checkpoint 1,161,776,976 bytes; cold first update
214.6 s (the 250 s phase time includes process start).

| microbatch | compile | median update s (steps 4–8) | targets/s | peak allocated |
|---:|---|---:|---:|---:|
| 16 | off | 0.862 | 152,036 | 14.16 GiB |
| **16** | **blocks** | **0.683** | **191,999** | 9.46 GiB |
| 16 | blocks + reduce-overhead | failed (same CUDA-graphs overwrite error as on H100) | — | — |

Beam list price read the same day: RTX 4090 0.000191667 USD/s (0.69 USD/h, shown "with committed
spend"), CPU 0.0000125 USD/s per core, RAM 0.0000021 USD/s per GiB; the launcher requests 4 cores and
32 GiB, so ≈ 1.11 USD/h all-in; the page also shows a 1.77 USD/h example for the same shape whose
basis is not stated. **Per 10⁹ tokens with the compile lane: 1.60–2.56 USD on the 4090 against 2.76
USD on the Modal H100 at microbatch 64; calendar 6.0 days against 2.9.** Beam functions run with
`timeout=-1`, so a 4090 run needs no 23-hour segments. The RTX 5090 arm did not obtain capacity on
the first attempt ("GPU capacity for RTX5090 is currently low"); a retry is pending. GPU cost of the
4090 arm ≈ 0.25 USD.

## Addendum — H100 pass 2 after the review fixes (source `a474340`+, same day)

- **Compiled production CLI, continuous and resumed (`--compile blocks`)**: both exit 0; the first
  post-resume loss is identical to the continuous run (step 4: 8.953735 / 8.953735) and the first
  resumed *update* differs by 2.0e-5 (step 5), inside the 7.5e-5 run-to-run band of the A10
  control. The compile lane's checkpoint/resume is therefore qualified to the same standard as
  eager. Compiled versus eager per-step losses stay at rounding level (step 6: 8.881857 vs 8.881739).
- **Backward autocast pinned to the eager contract** (`backward_pass_autocast="off"`, ADR 0180):
  throughput unchanged — 397,404 targets/s at microbatch 64 (397,698 before the pin), peak
  27.42 GiB; eager at microbatch 64 288,654. CUDA graphs fail as before.
- Modal cost of pass 2 ≈ 0.68 USD (617 s local, two containers). No containers left running.

## Addendum — 500 compiled updates on H100: Quantile Balancing beyond startup (same day)

Engine-level, `accepted()` untouched, `--compile blocks`, microbatch 64, BF16, constant LR 3e-4
(probe default; the schedule is not the object here), 500 updates = 65.5 M tokens of the real
SuperBPE corpus. Cost ≈ 0.55 USD (441 s plus a skipped first attempt).

| update | loss | grad norm | layer-0 bias max / std | peak allocated |
|---:|---:|---:|---|---:|
| 1 | 9.047 | 0.39 | 0.033 / 0.013 | 27.4 GiB |
| 20 | 8.310 | 1.63 | 0.260 / 0.037 | 26.0 GiB |
| 50 | 7.262 | 0.69 | 0.797 / 0.103 | 30.4 GiB |
| 100 | 6.497 | 0.25 | 0.762 / 0.100 | 30.1 GiB |
| 200 | 5.698 | 0.31 | 0.687 / 0.103 | 27.4 GiB |
| 300 | 5.103 | 0.43 | 0.664 / 0.113 | 28.0 GiB |
| 500 | 4.434 | 0.53 | 0.820 / 0.144 | 27.9 GiB |

- **Learning is real**: 9.05 → 4.43 in 500 updates. Gradient norm peaks at 1.6 around update 20
  (clipped at 1.0) and settles below 0.6.
- **The selection bias saturates instead of diverging**: it climbs during the first ~50 updates and
  then stays at 0.65–0.82 (std ≈ 0.10–0.14) on a score scale of order 1 — Quantile Balancing is
  carrying a stable, sizeable correction, not a runaway one.
- **Loads**: cumulative per-layer max load fraction 0.023–0.035 (uniform 0.0156, i.e. 1.5–2.2×),
  min 0.008–0.011, **zero dead expert slots** across 8 layers × 64 experts after 500 updates. The
  cumulative figure includes the unbalanced first updates; instantaneous balance is tighter.
- **Throughput is steady**: median 0.3296 s per update = **397,614 targets/s**, p95 0.338 s.
- **Memory has spikes**: per-update peak (reset every update) median 28.5 GiB, p99 34.4 GiB, one
  update (408) at **53.4 GiB** and 0.56 s — consistent with a Dynamo recompilation or a routing
  capacity spike (`capacity = max load` sizes the padded buffer). Harmless on 80 GB at microbatch
  64; on a 24 GB card the same 1.9× spike over a 9.5 GiB base (microbatch 16) still fits, a
  microbatch-32 base (≈ 19 GiB) would not. **Keep ≥ 2× headroom over the typical peak on the
  chosen GPU.**

## Addendum — corrections from the final external review (Astra high, same day)

- "Identical validation" holds at the printed 3 decimals only: continuous-vs-resumed validation loss
  differs by 7–8e-5 (H100 8.392e-5, Beam 8.297e-5, compiled H100 7.272e-5), the same order as the
  per-step CUDA noise.
- The A10 control shows the resumed step-5 loss slightly *below* both scratch runs; the claim is
  therefore "the resume difference (4.3e-5) is smaller than the scratch-to-scratch difference
  (7.5e-5)", not "inside the interval".
- Routing normalisation: the CLI telemetry divides per-layer counts by tokens (uniform 2/64), the
  probe's cumulative fractions divide by assignments (uniform 1/64); the 2.2× figure is on the
  latter. Bias saturation, unverified at 12 updates, is now measured over 500 (previous addendum).
- The 397,698 figure is the first pass; 397,404 is the pass with the pinned backward autocast and
  the one to quote.
- Modal bills host CPU/RAM on top of the GPU; the 4-core/32-GiB shape used by the qualification
  adds ≈ 0.44 USD/h, so the H100 compiled run is 276–307 USD, Beam 4090 161–287 USD depending on
  the tariff basis (1.11 USD/h from listed components, 1.77 USD/h example for a 2-core/16-GiB
  shape, ≈ 1.98 USD/h for this launcher's shape by the listed increments).
