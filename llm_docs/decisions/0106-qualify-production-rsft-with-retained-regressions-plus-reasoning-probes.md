---
status: accepted
date: 2026-08-20
supersedes: null
---

# 0106 — Qualify production R-SFT with retained regressions plus reasoning-specific probes

## Context and problem statement

The accepted production R-SFT R0 checkpoint `100m-2b-rsft-r0-12306-001` extends the completed S0 parent with the atomic `<think>`, `</think>`, `<answer>` interface and a 50,260-token semantic vocabulary. The existing post-SFT qualification is not sufficient by itself: it assumes identical model geometry, does not understand the reasoning-token protocol, and does not measure whether the new reasoning behavior transfers to novel problems.

At the same time, replacing the established S0 matrix with only reasoning benchmarks would hide base-language or instruction-retention regressions. Evaluation therefore needs a direct S0-to-R-SFT comparison that preserves the existing regression tests and adds reasoning-aware measurements.

Recent reasoning-model practice also argues against one deterministic CoT score. DeepSeek-R1 evaluates sampling-sensitive reasoning with repeated stochastic generations, self-consistency work shows that greedy CoT can understate or misrepresent reasoning behavior, and 2025–2026 faithfulness work shows that plausible visible CoT should not be treated as a reliable certificate of the model's internal causal reasoning. At sub-1B scale, 2026 results also show strong capacity effects on difficult mathematical reasoning, so competition-math floors are not an appropriate primary gate for this 100M R0 stage, whose frozen scope is seven self-contained logic primitives rather than math/code.

## Considered options

- Reuse `post_training.sft.eval_suite` unchanged and ignore the R-SFT vocabulary/protocol difference.
- Replace the S0 qualification with public CoT/math benchmarks such as AIME or ProcessBench.
- Use an LLM judge to grade the natural-language reasoning trace.
- Preserve the S0 regression matrix and add a mechanically scored reasoning/protocol layer specialized to the frozen R0 skill contract.

## Decision outcome

Chosen option: **preserve the S0 regression matrix and add a mechanically scored R-SFT reasoning layer**, because it gives a clean incremental S0→R-SFT measurement without conflating answer correctness, protocol compliance, and speculative CoT faithfulness.

The canonical full R-SFT qualification is frozen as follows:

1. Compare completed S0 `100m-2b-sft-s0-001` directly against production R-SFT `100m-2b-rsft-r0-12306-001`.
2. Retain `eval_core_v1` on both checkpoints.
3. Retain the completed S0 validation/test masked-loss suite on both checkpoints as an instruction-retention check.
4. Retain the 30 deterministic S0 instruction-behavior probes. R-SFT answers are scored with the same mechanical acceptance rules after extracting the final `<answer>` segment; reasoning-token protocol quality is reported separately.
5. Retain the full 18-prompt greedy regression at temperature `0`, top-p `1`, top-k `0`, seed `17`, maximum 32 new tokens.
6. Retain the full 18-prompt wider regression at temperature `1.0`, top-p `0.9`, top-k `20`, seed `17`, one sample, and each prompt's native generation budget.
7. Add R-SFT production-bundle validation/test masked loss for held-out post-R-SFT data.
8. Add a frozen novel 35-case reasoning generation suite: five mechanically scored cases for each R0 skill `INF`, `DED`, `REL`, `CSP`, `IND`, `ABD`, and `MAG`. The cases are self-contained and are not copied from the production R-SFT corpus.
9. Run the novel reasoning suite twice on S0 and R-SFT:
   - deterministic diagnostic: temperature `0`, top-p `1`, top-k `0`, seed `17`, maximum 256 new tokens;
   - sampled pass@1 estimate: temperature `0.6`, top-p `0.95`, top-k `0`, eight responses per problem, maximum 256 new tokens. Eight is a practical Small-LLM qualification default; callers may increase it for tighter publication estimates.
10. For R-SFT, report atomic protocol metrics separately: exactly one reasoning start, reasoning end, and answer start token; correct marker order; non-empty reasoning and answer; EOS/runaway rate; and reasoning/answer token lengths.
11. Final-answer correctness and protocol compliance remain separate axes. A plausible CoT is not scored as faithful merely because it reads well, and no LLM-as-judge trace score is part of the canonical gate.
12. Do not collapse the qualification into one master score. Interpret reasoning acquisition, instruction following, protocol health, S0 retention, and base-language retention together.

The canonical Kaggle entry point is:

```bash
python kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite full
```

## Consequences

### Positive

- R-SFT can be compared directly with the completed S0 checkpoint despite the semantic-vocabulary extension.
- The historical pretraining/S0 regression matrix remains comparable across stages.
- Reasoning gains are measured on held-out data and novel mechanically scored problems rather than only on training examples.
- Greedy and sampled reasoning behavior are both visible.
- Protocol failures cannot hide behind correct-looking answers, and correct answers are not incorrectly promoted into claims of CoT faithfulness.
- The suite is scale-appropriate for a 100M model and aligned with R0's frozen reasoning scope.

### Negative or limiting

- The 35-case novel reasoning suite is intentionally compact and is not a general-purpose reasoning leaderboard.
- Eight samples per problem are a practical pass@1 estimate, not the 16–64 sample budgets used by large reasoning-model papers.
- The suite does not establish causal faithfulness of visible CoT.
- Public math/code benchmarks remain useful later, but are not primary R0 acceptance gates.

## Validation

A canonical full run must produce one versioned report containing S0 and R-SFT scorecards, S0→R-SFT deltas, both historical qualitative sampling contracts, the novel reasoning greedy and sampled results, and R-SFT protocol metrics. The launcher must fail closed on a non-production R-SFT token contract or incompatible model geometry other than the frozen semantic-vocabulary extension.

## Links

- `llm_docs/runbooks/rsft_r0_qualification.md`
- `llm_docs/decisions/0082-focus-r0-on-logic-primitives-and-defer-exact-computation.md`
- `llm_docs/decisions/0084-qualify-sft-checkpoints-with-standard-model-evaluation-matrix.md`
- `llm_docs/decisions/0099-freeze-production-rsft-special-token-interface.md`
- `llm_docs/decisions/0105-accept-production-rsft-r0-run.md`
- DeepSeek-R1 model card evaluation protocol
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- Muennighoff et al., *s1: Simple test-time scaling*
- Zhuang et al., *Effective Learning for Small Reasoning Models: An Empirical Study on 0.5B Reasoning LLMs* (2026)
- Mittal and Arike, *C2-Faith* (2026)
