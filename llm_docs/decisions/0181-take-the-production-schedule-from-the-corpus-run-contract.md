---
status: accepted
date: 2026-09-12
supersedes: null
---

# 0181 — Take the production learning-rate schedule from the corpus run contract

## Context and problem statement

The accepted production launcher (ADR 0177) built the `python -m MOE_model` command without
`--schedule`, `--warmup-tokens`, `--stable-tokens`, `--decay-tokens` or `--minimum-lr-ratio`
(verified: zero matches in `moe_production.py` and both provider launchers at `0bc4acb`). The
trainer's default is `--schedule constant`, so the 100 B-token run would have trained at a flat
3e-4 with no warmup and no decay, and the defect would have surfaced only at the end. The corpus
producer already writes the intended schedule into `run_contract.json` (`trainer`: WSD, 5 %
warmup, 75 % stable, 20 % decay, floor 0.1, 762,940 updates for the live 100 B corpus), and the
paired pilot read exactly those fields. The launcher also relied on trainer defaults for the
Muon recipe the pilot pins explicitly, and never validated because `--validation-blocks`
defaulted to 0 and periodic checkpoints do not trigger validation on their own.

## Considered options

- Hard-code the 100 B schedule in the launcher.
- Read the schedule from the corpus `run_contract.json`, deriving the standard WSD plan when a
  finite retokenized corpus carries only its block geometry, and fail closed when neither exists.
- Leave the schedule to the operator through new launcher flags.

## Decision outcome

Chosen option: **read the schedule from the corpus run contract**, because the corpus and the
schedule are produced together and the pilot already did this; a hard-coded value would drift
from the corpus, and operator flags reintroduce the silent default. `moe_production.resolve_schedule`
returns the contract's `trainer` plan or `standard_wsd_plan` from `planned_train_blocks`;
`build_training_command` refuses a `total_steps` beyond the plan, passes the schedule and the
pinned optimizer recipe (`--learning-rate 3e-4 --weight-decay 0.1 --muon-momentum 0.95
--muon-lr-multiplier 1.0 --muon-update-rms 0.18 --muon-weight-decay 0.1 --max-grad-norm 1.0`),
validates every checkpoint (`--evaluation-every-steps` = checkpoint cadence) with the contract's
validation blocks unless the operator passes an explicit count, and the provider result payload
records the schedule used. Launcher dry-run displays use a placeholder because the data volume
is not mounted locally.

## Consequences

### Positive

- The production run trains the recipe the corpus was planned for; the schedule cannot silently
  fall back to a trainer default.
- The best-checkpoint promotion and retention (ADR 0179) now see a validation metric every
  checkpoint.

### Negative or limiting

- A dataset directory without `run_contract.json` cannot be launched; the qualification corpus
  needs its contract staged next to the shards.
- Validation every 1,000 updates costs 16 validation blocks each time (seconds on the measured
  GPUs), accounted in the calendar, not the update clock.

## Validation

`tests/test_moe_production_schedule.py`: the live contract's values reach the command verbatim;
a geometry-only contract derives the same plan as the producer; a missing contract or a
non-WSD plan fails closed; `total_steps` beyond the plan is refused; the trainer parses the
emitted command. A CPU dry run of the built command on the staged qualification corpus must
report `schedule: wsd` in its identity.

## Links

- `0177-accepted-moe-production-launch-contract.md`
- `0179-production-checkpoint-sequence.md`
- `dataset/incremental_frontier.py` (`standard_wsd_plan`, `build_run_contract`)
- `/home/edo/Documents/0_Projects/Small-LM/docs/investigations/2026-09-12-checkpoint-e-pronti-al-training.md` §8
