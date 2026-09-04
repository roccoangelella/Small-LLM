# Current Small-LLM Project Status

Last reviewed: 2026-09-04

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
