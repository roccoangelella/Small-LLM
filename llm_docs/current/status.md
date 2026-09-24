# Current Small-LLM Project Status

Last reviewed: 2026-09-23

Verification: [CPU provider continuation](../evidence/moe_provider_continuation_cpu_2026-09-13.md).

Runtime-efficiency candidate: explicit source-origin/executor separation preserves
v1 receipt rollback; validation batching and single-worker asynchronous upload are
opt-in. Active Modal H100 executor is97af6ea (edo/modal-durable), scientific origin556f4f7. Same run resumed HF50000 through durable spawn; launcher exited while GPU advanced beyond51100. Remote ownership/data guards are active; local observer is optional. First subsequent remote checkpoint verification is recorded in the hub adoption lane. Async upload remains OFF, validation microbatch1.
[Controls, tests and adoption limits](../runbooks/runtime-efficiency.md).

## Current accepted MoE work

The production path is the accepted E64/Top-2, L8/d256/h352 GDN-2 hybrid with
SuperBPE 8,000 IDs (8,192 physical rows), BF16 and compiled blocks. The 2026-09-12
qualification includes the real production CLI, checkpoint/resume, Modal container
replacement, Beam streaming and 500 compiled H100 updates; see the final addenda in
[the qualification evidence](../evidence/moe_production_qualification_modal_2026-09-12.md).
Those are short qualifications, not a completed 100B trajectory.

ADR 0183 now wires the MoE launchers to HF latest/best checkpoint durability and
one stable W&B identity across accounts. The accepted loader allows microbatch and
observation cadence changes while preserving the scientific recipe and data cursor.
`--resume latest` fails if neither local nor remote progress exists; `--resume new`
is required for first launch. Offline tests cover real tiny-MoE continuation through
the actual CLI and byte-accurate bucket/W&B fakes. Live cross-GPU migration with this
integration has not been run. [Procedure](../runbooks/moe-provider-continuation.md).

Kaggle 2×T4 read-only continuation qualification (2026-09-24): the live run 003
checkpoint/corpus restored and passed the CPU gate. A MoE-specific two-rank
adapter replayed the full 64-sequence block across both GPUs; offline CLI
validation/save and a second exact local resume passed on the public
`moe-8e-top1` checkout (`7ed4e29`), with global router quantiles and
rank-zero-only side effects. Two isolated 16-step Kaggle W&B runs measured
**15.3k → 18.2k warmed-up targets/s (+18.9%)** when the safe microbatch
increased from 1 to 4; both validated/saved locally without HF publication.
T4 has **no native BF16**, and the H100 writer reported ~346k with compiled
blocks and microbatch 32; do not promise hardware parity. The original W&B
writer was `running` during the isolated tests but last queried as `crashed`;
its actual owner/process and latest HF pointer are not reconciled. Live Kaggle
W&B resume and remote publication remain **unqualified**. Do not start another
writer without confirming the old process has stopped and rerunning preflight.
[Kaggle runbook](../runbooks/moe-kaggle-continuation.md) · [W&B test](../evidence/2026-09-24-moe-kaggle-wandb-perf-test.md) · [Optimization](../evidence/2026-09-24-moe-kaggle-mb4-optimization.md).

The authorized 100B run is active on Modal after an OOM recovery at microbatch32;
remote checkpoints through32500 were observed by the local supervisor on September13.
The runtime-efficiency candidate has not been deployed. Current account, checkpoint,
corpus hold and billing observations live in the Small-LM project hub monitoring lane.
No additional GPU experiment was launched for these runtime changes.

Corpus discussion is next: the current producer excludes programming cluster 11
but retains mathematics clusters. It does not guarantee conversational-only or
code-free text. Tokenizer provenance is ADR 0175: 10 GB decoded from the existing
10B-token qualification corpus, with stage 2 on a 2 GB prefix. No tokenizer or
dataset recipe is changed by the continuation work.

MoE local chat: the live best pointer for `moe-100b-superbpe-003` needs
`tokenizer/superbpe_8000_v2.json`, not run 001's `superbpe_8000.json`.
A same-checkpoint, same-text test scores 3.074 CE with v2 versus 10.569 with v1;
run 001 reverses that relationship. Chat now pins and hashes the tokenizer per
known run and rejects unknown 8k identities. This fixes the gibberish, **not**
pretrained-model instruction following: greedy `hello` under the SFT-style chat
template still loops on `Assistant:`. Dataset manifest tokenizer identity is not
yet checked by the trainer. [Measured evidence](../evidence/moe_chat_tokenizer_mismatch_2026-09-23.md).

## Historical detail through 2026-09-11

The dated material below records earlier stages; its old MoE blockers/priorities
are superseded by the current section above and ADRs 0173–0183.

MoE branch execution efficiency (2026-09-10, local commits `b5e3ffb`, `8f195ee`, `fbe4ef2`, not pushed): ADR 0170 batches Muon Newton–Schulz by matrix shape, ADR 0171 fuses attention (SDPA) and scores the output loss in 4,096-token chunks with recompute, ADR 0172 sorts MoE dispatch with one host sync per layer. Equivalence to the previous implementations is proven on CPU (bit-identical for Muon and dispatch; FP32-rounding tolerance for attention and loss) by 18 new tests; the full suite (623 tests) shows no failing test id that was not already failing in the 2026-09-08 baseline log. Measured baseline motivating the work: [RTX 4090 profile](../evidence/moe_execution_profile_rtx4090_2026-09-08.md) — 11,073 targets/s, MFU ≈ 4 %, 521,704 launches and 5,120 `nonzero` syncs per update. **Measured on GPU the same day** by a paired A/B on one Modal A10 (same container, same data, same seed, code tree the only variable): **+19.6 % warm throughput** (8,990.7 → 10,750.0 targets/s), **−1.89 GiB peak** (14.31 → 12.43 GiB), **−21 % kernel launches**, **−66 % host synchronizations**, `nonzero` eliminated. The freed memory admits microbatch 4 for a further +13.2 % (**+35.3 % total**), which the old code cannot run: it dies on the 1.54 GiB FP32 logits tensor the chunked loss removes. The old arm reproduces the 2026-09-08 A10 figure to 0.5 %, validating the measurement chain. [Evidence](../evidence/moe_execution_ab_modal_a10_2026-09-10.md); contracts and remaining priorities in [`training_execution_efficiency.md`](../reference/training_execution_efficiency.md). Vast was unusable for this (no ssh from the team account).

MoE branch: the paired M0/M1 pilot is governed by [ADR 0168](../decisions/0168-moe-paired-pilot-controller-and-observation.md) and the [pilot runbook](../runbooks/moe-paired-pilot.md); H100 single-update qualification passes for D/M0/M1 at microbatch8 in FP16/BF16; M0 microbatch16 fails with OOM in BF16. [Measured scope](../evidence/2026-09-08-modal-moe-qualification.json). Scientific pilot results remain unavailable.

MoE 100B SuperBPE dataset production is active on `moe-8e-top1` via a direct VPS process started 2026-09-11 11:31 UTC in tmux session `moe-100b-superbpe-public` (PID `3344160`), running `python -m dataset.moe_100b` with output under `/tmp/small-llm-moe-dataset/producer`. The earlier Modal app `ap-h4isK62Lffpz579Ea4lZ5B` stopped with zero tasks. The run targets `moe-100b-superbpe-b64-dataset-001` in public bucket `roccoangelella/small-llm-moe-100b-superbpe-dataset`; source resolution remains 100 files / 1,851.4 GiB. This is dataset production only and does not authorize 50B/100B-token model pretraining. Completion remains gated on ADR 0178 visibility, READY-frontier, coverage, manifest, and hash checks.

## Repository and protocol state

- Repository: `roccoangelella/Small-LLM`.
- Current evaluation decisions: ADR 0140 defines evaluation v2 and ADR 0141 activates the pretrained and SFT evaluator entrypoints.
- ADR 0142 ignores local agent-tooling directories (`.agents`, `.Agents`, `.pi`) without deleting their already tracked historical contents.
- ADR 0143 removes IRE project state (`.ire/`) from the repository and ignores the directory going forward; `llm_docs/` is the canonical project-memory system.
- ADR 0144 consolidates current 100M/10B post-completion pretraining diagnostics in `kaggle/src/probes_100m_10b.py`; the active branches hold LR at `1e-5` and `2e-5` for 3,000 updates from the preferred step-71,750 source, with a strict same-repository current-best fallback when that artifact is unavailable.
- ADR 0145 synchronizes active README lifecycle wording with this status file and requires completed-run procedures to be labeled as reproduction/history rather than current launch authorization.
- ADR 0146 makes `kaggle/probes_100m_10b.py` the stable operator entrypoint and normalizes repository, `kaggle/src`, `beam/`, and cached `runtime` module paths before delegating to the ADR-0144 implementation. This fixes the post-`src/`-move runtime error that incorrectly expected `kaggle/beam/runtime.py`.
- ADR 0147 keeps all evaluation-v2 benchmark cases and scoring contracts unchanged while batching L20 conditional-likelihood requests, Base Prompt v2 generation, and SFT Behavior v2 generation. L20 is length-bucketed with a 16-request / 8,192-padded-token cap; generated views use per-request RNG generators and a 16-request batch cap. Coarse progress reporting is now mandatory for these long phases.
- ADR 0148 registers the completed `100m-10b-sft-s0-2b10pct-data-001` trajectory as the `(100M, 10B)` SFT default in `chat.py`; `python chat.py --model_params 100M --num_tokens 10B --sft` now uses the existing fail-closed SFT checkpoint loader, while the `(100M, 10B)` pretrained chat profile remains unregistered.
- ADR 0149 corrects the Base Prompt v2 construction bug: the active full set now contains 120 unique prompt texts and IDs, with exactly 20 unique objective prompts in each of the five objective families and 20 unique qualitative prompts. Older recycled-template Base Prompt v2 aggregates are historical defective evidence and must not be interpreted as a 100-unique-prompt statistic.
- ADR 0150 removes local substring/regex scoring from Base Prompt v2. The GPU evaluator now emits raw prompt/reference/continuation evidence with pending judge status; `trainer.base_prompt_judge` scores the 100 objective cases afterward through the same GemRouter endpoint used by R-SFT. Greedy and sampled views are judged semantically; the 20 qualitative cases remain unscored.
- ADR 0151 makes `chat.py` persist downloaded local-chat artifacts under root-level `chat_models/<stage>/<run_id>/`, re-verify cached checkpoints before reuse, ignore that runtime cache in Git, and print the effective generation configuration at startup without changing the current sampling values.
- MoE pretrained chat profile: `python chat.py --moe` checks `run/moe-100b-superbpe-003/best.json` in the configured HF checkpoint bucket on every launch, reuses the verified local cache when the pointer is unchanged, and downloads/verifies a new best before replacing the cache when it moves. A missing or unreachable pointer is an error, not permission to use stale weights. The previous pinned step-75,000 snapshot remains explicitly available with `python chat.py --moe --run-id moe-100b-superbpe-003-chat-best-step-00075000` (`latest.json` in that separate run); it does not track future bests. Chat dynamically resolves `MoESmallLLM`, uses the pinned 8,000-token v2 SuperBPE tokenizer (`tokenizer/superbpe_8000_v2.json`, EOS 7,992), and accepts these intermediate checkpoints without `--allow-incomplete`; explicit `--run-id moe-100b-superbpe-001 --pre-trained` remains available and uses the v1 tokenizer. The interactive `chat.py` loop treats every input as an independent raw completion prefix with the same sampling seed, and streams the continuation without role labels or chat history (for all stages, including SFT). There is no `/clear` command; only `/quit` and `/exit` have control behavior. Synthetic param/token pairs like `(100M, 100B)` are not registered.
- `small-llm-eval` / `trainer.eval_entrypoint` route pretrained checkpoint evaluation through `trainer.eval_suite_v2`.
- `post_training.sft.eval_suite` is now the v2 SFT qualification entrypoint, so existing SFT launchers keep their module path while emitting v2 JSON.
- SFT Behavior v2 is the primary instruction-following suite; the legacy 30-case behavior suite remains in the JSON only as `instruction_behavior_v1_legacy`.
- Canonical sampled qualitative decoding target is `temperature=1`, `top_p=1`, `top_k=0`.
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

The first Kaggle execution of `100m-10b-sft-s0-2b10pct-data-001` stopped because
the available T4 session time was exhausted. That W&B `failed` state was an
infrastructure interruption, not an SFT-quality failure. The restarted job
completed on 2026-09-04. W&B reports the run as `finished` at global step 6,219,
with 200,099,738 consumed loss-bearing SFT targets and final validation loss
1.4850977542185604.

The completed trajectory is now registered for local chat under ADR 0148:

```bash
python chat.py --model_params 100M --num_tokens 10B --sft
```

This path uses the normal GPT-2 tokenizer, the existing SFT chat template, and
the existing verified-completed-checkpoint gate. No 100M/10B pretrained chat
profile is registered.

### Active SFT qualification

The active SFT evaluator now emits `small-llm-post-sft-qualification-v2`.
Its first sections are `read_me_first` and `headline_summary`, followed by
checkpoint metadata, `eval_core_v1`, masked SFT validation/test loss,
`instruction_behavior_v2`, `instruction_behavior_v1_legacy`, and
`base_prompt_suite_v2`.

SFT Behavior v2 is the primary instruction-following evaluation:

- 180 semantic tasks;
- six balanced families;
- L0 capability plus L1/L2/L3 progressively constrained variants;
- 720 total cases;
- 480 diagnostic cases;
- 240 held-out qualification cases;
- conditional compliance only where the same task's L0 semantic answer is correct;
- greedy primary plus sampled robustness over seeds 17/18/19;
- paired parent/SFT wins, losses, ties and exact McNemar statistics.

Behavior v2 generation is length-bucketed and evaluated in batches of up to 16 independent requests while preserving each case/seed identity and restoring the original output order. Base Prompt v2 uses the same evaluation-only batching helper. ADR 0149 changes the Base Prompt v2 prompt definitions only to remove recycled cases; its decoding parameters, native generation budgets and batching semantics are unchanged.

Under ADR 0150, Base Prompt sections in parent/SFT scorecards are raw unjudged evidence. Their local accuracy fields are therefore absent or null rather than string-matched. `trainer.base_prompt_judge` accepts the full parent-versus-SFT qualification JSON and emits separate semantic judgments for parent and SFT using one judge contract. SFT Behavior v2 scoring is unchanged.

## R-SFT

The production R-SFT path remains atomic-protocol based and retains its
reasoning-specific qualification. The evaluation v2 target is for base
qualitative regressions to use native prompt budgets and for the general sampled
view to follow `temperature=1`, `top_p=1`, `top_k=0`. Reasoning pass@1 keeps its
own task-specific sampling protocol.

## Active pretraining evaluation v2

The active pretrained checkpoint entrypoint emits
`small-llm-pretraining-evaluation-v2` through `trainer.eval_entrypoint`.
Its first sections are `read_me_first` and `headline_summary`, followed by
checkpoint metadata, frozen `eval_core_v1`, the external L20-style suite, and
the expanded base prompt suite.

Canonical full pretraining qualification now consists of:

1. frozen `eval_core_v1`;
2. six-task L20-Edu-style zero-shot conditional-likelihood evaluation using
   `lm-evaluation-harness==0.4.12`;
3. the corrected Base Prompt v2 set `base-prompt-v2-unique-120-2026-09-04`, with
   100 unique objective cases (20 per objective family) and 20 unique readable
   qualitative continuations.

The active Base Prompt v2 constructor fails closed if any case ID or prompt text is duplicated or if the 100/20 and per-family counts drift. The GPU evaluator does not assign local pass/fail verdicts to the 100 objective cases. It records `reference_answer`, raw continuation, token evidence, and `judge_status=pending`; the separate GemRouter postprocessor produces semantic correctness and per-family/overall accuracy. Results produced by either the earlier recycled-template implementation or the old substring/regex scorer are not canonical Base Prompt semantic-judge scores. This correction does not affect `eval_core_v1` or L20 results.

GemRouter Base Prompt judgment uses the R-SFT endpoint/auth contract (`GEMR_API_KEY`, `LLM_ENDPOINT`), requests `gemini-3.7-flash` by default at temperature 0, requires the Gemini-only/no-fallback health state, batches up to 20 cases by default, retries malformed/provider failures, and records source/judge provenance. Base Prompt scores are comparable only when prompt-set ID, judge model, and judge prompt ID/hash match.

The six external tasks are ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA OpenAI,
PIQA and WinoGrande. Full qualification should use all available benchmark examples.
Fast mode may limit the external tasks and is diagnostic only.

L20 execution no longer performs one model forward per answer candidate. Requests are encoded exactly as before, sorted into similar-length batches, capped at 16 requests and 8,192 padded input tokens, scored together, and restored to harness order. The evaluator prints coarse request-completion progress. The benchmark sample count and metric contract are unchanged.

The current optimization intentionally does not use `torch.nn.DataParallel`; dual-T4 evaluation remains a future DDP/`torchrun` qualification task consistent with the repository's established dual-GPU architecture.

## Evaluation dependency boundary

`lm-evaluation-harness==0.4.12` is pinned in `requirements-eval.txt` rather
than the training lock. Evaluation tooling must not perturb the training
environment.

Cache integration (2026-09-08): ADR0169 now wires reviewed Triton seed restore/harvest into both pilot wrappers. Local lifecycle tests cover failure and publication paths; GPU cache-hit/durability/savings checks are pending the next authorized useful run. No dedicated cache build or new GPU run was launched.


## Durable Modal continuation candidate — 14 September2026

Remote spawn and atomic run/attempt claims plus remote HF corpus hold implemented; numerical trainer unchanged from runtime candidate9e31d88. Not yet adopted at this documentation snapshot. See [runbook](../runbooks/modal-durable-continuation.md). Live source/checkpoint/app and adoption evidence are tracked in the Small-LM project hub run lane.
