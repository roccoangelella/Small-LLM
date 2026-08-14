---
status: accepted
date: 2026-08-14
supersedes: 0050
---

# 0071 — Run the full fresh 100M / 10B trajectory while evaluating 5B concurrently

## Context and problem statement

The exact ADR-0025 comparison is complete. The 100M/2B endpoint ties the
20M/2B endpoint at 2/12 strict greedy QA answers, but its generations stop
before the 32-token cap much more often, are markedly less repetitive, expose
more facts under matched sampled decoding, and improve every retained intrinsic
cluster and context-position bucket. The deterministic 10B corpus is also
complete and verified in both its authoritative HF bucket and Beam cache copy.

ADR 0050 authorized a fresh 10B trajectory conditionally, with a 5B evaluation
acting as a pause/continue gate. The user now wants the true full trajectory to
keep training while the 5B checkpoint is evaluated separately on Kaggle.

## Considered options

- Stop the Beam session near 5B, evaluate, and launch a second segment only
  after an explicit continuation decision.
- Launch the full 10B plan once and evaluate the nearest durable 5B checkpoint
  concurrently on Kaggle.
- Skip the intermediate evaluation and inspect only the final endpoint.

## Decision outcome

Chosen option: **launch the fresh `100m-10b-data-001` trajectory for its full
76,294-update / 10,000,007,168-target plan and run the approximately-5B Kaggle
evaluation concurrently without pausing or terminating Beam training.**

The completed exact behavioral evidence is accepted as sufficient to close the
fresh-launch gate. The launch must omit `--max-steps-this-session`, retain fresh
initialization, use the completed `modal-10b-b64-dataset-001` corpus, and keep
the frozen 100M geometry, seed, context, optimizer block, precision, GDN-2
backend, and standard WSD schedule.

The intermediate Kaggle evaluation remains scientifically useful, but it no
longer authorizes or blocks continuation to 10B. Because live HF checkpoints
use rolling latest-only retention every 500 updates, the preferred midpoint
artifact is `step-00038000` / 4,980,736,000 consumed targets. Its exact identity
must be captured for Kaggle before the live pointer advances.

## Consequences

### Positive

- The experiment measures the intended terminal 10B WSD trajectory without a
  provider pause or a second launch boundary.
- Kaggle evaluation can overlap paid training time and still provide a useful
  approximately-5B behavioral snapshot.
- The launch remains a clean fresh 100M data-scaling comparison.

### Negative or limiting

- A poor 5B result will not stop the already-running trajectory, so this choice
  gives up ADR 0050's compute-saving escape hatch.
- `step-00038000` is an intermediate stable-phase checkpoint, not a terminal 5B
  WSD endpoint.
- Rolling checkpoint retention creates a bounded capture window for Kaggle
  staging unless the midpoint checkpoint is copied to an isolated namespace.

## Validation

- The Beam launch contract reports `max_steps_this_session = remaining plan`.
- The trajectory starts from no prior `run/100m-10b-data-001` checkpoint.
- Periodic local/HF checkpoints remain exact-resume capable through the run.
- Kaggle evidence records the exact midpoint checkpoint ID and consumed-target
  count while Beam training continues.
- The terminal checkpoint reaches update 76,294 and receives the ordinary full
  post-pretraining qualification.

## Links

- [`../evidence/scaling/100m_2b_behavioral_qualification_2026-08-13.md`](../evidence/scaling/100m_2b_behavioral_qualification_2026-08-13.md)
- [`../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md`](../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md)
- [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md)

