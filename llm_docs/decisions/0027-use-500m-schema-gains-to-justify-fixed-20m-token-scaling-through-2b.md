---
status: accepted
date: 2026-08-10
---

# 0027 — Use 500M qualitative schema gains to justify fixed-20M token scaling through the planned 2B probe

## Context and problem statement

The canonical deterministic full post-pretraining prompt suite was run on the completed 20M / 500M checkpoint (`step-00015264`, 500,156,416 consumed training target tokens). The checkpoint still fails direct factual retrieval in the twelve simple Q/A probes, but its generations show clearer learned text schemas and discourse formatting than the earlier smoke-scale evidence.

Notable 500M behaviors include:

- `Question: ... Answer:` prompts are continued with answer-shaped declarative text and often another correctly formatted Q/A record, even when the factual answer is wrong;
- the Alice/Ben dialogue probe preserves speaker alternation, emitting an Alice turn followed by a `Ben:` turn;
- the sentiment-pattern probe emits a plausible `negative` label before later degeneration;
- the model continues structured and topical surface patterns despite weak entity/relation binding;
- validation loss had continued to improve over the scaling trajectory rather than showing an obvious qualitative/optimization saturation signal at 500M.

The earlier approximately-10M qualitative checkpoint had already shown some Q/A surface-format imitation, so the 500M evidence should not be described as the first appearance of the schema. The stronger claim supported here is that schema continuation, answer-shaped generation, and dialogue-format preservation are more developed while the model remains far from reliable factual or semantic performance.

## Decision outcome

Treat the 500M qualitative result as **positive evidence that additional pretraining tokens are still producing useful capability/structure gains at the fixed approximately-20M parameter scale**, and therefore continue with the already-authorized fresh 20M / approximately-2B-token scaling probe before increasing model size.

For this scaling step:

- keep the 20M model geometry fixed;
- keep the approved data/tokenization/training recipe fixed except for already-authorized runtime/backend changes;
- train the fresh 2B trajectory as a controlled data-scaling point;
- rerun the canonical full qualitative suite and frozen quantitative evaluation stack at 2B;
- compare schema adherence, factual relation binding, repetition, semantic coherence, validation loss, and teacher-forced confidence/rank against the 500M baseline.

This decision does **not** claim that quality will improve indefinitely with more tokens at fixed model size. The 2B experiment is used to measure whether the gains continue, diminish, or saturate as the token/parameter ratio rises substantially.

## Rationale

The 500M result is materially more encouraging than a raw `0/12` factual score suggests. A pretrained causal model learning that a `Question:` field should be followed by an `Answer:`-shaped statement, that dialogue labels should alternate, and that a sentiment field should contain a class-like token demonstrates learned conditional structure beyond generic local English fluency.

These are still weaker capabilities than factual retrieval or robust reasoning, but they are exactly the kinds of intermediate behaviors expected to become clearer before a tiny base model becomes reliably useful. Because the held-out objective also continued improving, there is no current empirical reason to stop the fixed-size data-scaling experiment at 500M.

Keeping model size fixed through 2B also preserves experimental clarity: if the 2B checkpoint improves over 500M under the frozen evaluation protocol, the gain can be attributed primarily to additional data/training rather than to parameter scaling.

## Guardrails against overinterpretation

- Q/A **format understanding is not equivalent to factual knowledge**. The 500M checkpoint often restates the relation without supplying the requested entity.
- Alice/Ben alternation is evidence of dialogue-schema continuation, not yet evidence of robust dialogue-state tracking; the actual content was repetitive and contextually weak.
- The sentiment output is encouraging but is one qualitative case, not a classification benchmark.
- Greedy repetition and semantic drift remain severe.
- A single 500M point cannot establish the eventual asymptote of a 20M model.

## Validation criterion for the 2B point

The decision to keep scaling tokens beyond 2B at the same parameter count should be revisited after the 2B run using the frozen comparison stack. Evidence for continued fixed-size scaling would include a meaningful combination of:

1. lower held-out loss/perplexity;
2. improved true-token rank/confidence in teacher-forced diagnostics;
3. stronger direct factual/relation completion;
4. better structured-pattern completion;
5. later or less severe greedy repetition;
6. improved semantic coherence while retaining schema adherence.

If those gains flatten materially while optimization remains healthy, parameter scaling becomes the more informative next axis.

## Links

- [`../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)
- [`0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- [`0025-freeze-canonical-full-post-pretraining-prompt-suite.md`](0025-freeze-canonical-full-post-pretraining-prompt-suite.md)
- [`../current/status.md`](../current/status.md)
