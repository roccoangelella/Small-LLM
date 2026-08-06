# Project overview

_Last reviewed: 2026-08-06_

## Goal

Build a modern dense decoder-only English language model below 1B parameters from random initialization as a serious learning and research project.

The intended system should eventually:

- produce coherent English text;
- acquire useful general knowledge and basic reasoning during pretraining;
- become conversational and instruction-following after separate post-training;
- use modern small-model architecture, optimization, evaluation, and serving ideas;
- remain reproducible enough that architectural and scaling claims can be tested rather than guessed.

Coding capability is not an initial target. The explicit programming cluster is excluded, although incidental code can remain because source clusters are imperfect.

## Development strategy

The codebase defines a geometry-scalable model family rather than one final model:

1. approximately 20M parameters for correctness, integration, and data-scaling experiments;
2. a first larger model only after the 20M/100M result is evaluated;
3. controlled intermediate sizes when evidence justifies them;
4. a near-1B model as a long-term goal, not an immediate run.

The initial context is 2,048 tokens. Longer context and additional architectures remain deferred until the base system is understood.

## Current architecture

```text
[GDN-2, GDN-2, GDN-2, gated full attention] x N
```

The approximately-20M geometry uses eight layers. The approximately-100M geometry is defined but not yet authorized for training. Detailed geometry and parameter accounting live in `model_geometry.md`.

## Dataset strategy

The initial source is the pinned GPT-2-tokenized Nemotron-ClimbMix revision. The pipeline:

- accepts clusters 1-10 and 12-20;
- excludes cluster 11, the explicit software/programming cluster;
- preserves the measured conditioned source-token mixture;
- assigns validation documents by a stable identity hash;
- packs context-plus-one sequences;
- writes immutable verified shards;
- supports interruption, resume, remote durability, and migration.

Google Drive is the durable dataset mirror, not the random-access training filesystem. Private Hugging Face storage is used for verified model/checkpoint publication. Kaggle T4 is the current training venue.

## High-level system

```text
pinned source corpus
      -> deterministic dataset preparation
      -> locally durable immutable shards
      -> bounded trainer consumer
      -> geometry-scalable hybrid decoder
      -> versioned joint model/data checkpoints
      -> frozen intrinsic and qualitative evaluation
```

## Resource assumptions

- Initial accelerator: one NVIDIA T4.
- Local/VPS storage: enough for bounded live cache, checkpoints, and publication staging.
- Durable dataset storage: personal Google Drive.
- Long-run first-pass corpus envelope: approximately 80B-100B accepted source tokens, subject to later authorization.
- Training must remain pausable and safely resumable across machines and Kaggle accounts.

## Documentation policy

`llm_docs/` is the project system of record, organized by purpose:

```text
current/    verified present state and roadmap
decisions/  numbered ADRs
reference/  detailed technical contracts
runbooks/   operational procedures
research/   investigations and external comparisons
evidence/   completed measured results
archive/    superseded plans and scaffolding
```

Use `../current/status.md` for present facts and `../decisions/README.md` for durable choices. Do not silently rewrite accepted evidence or decision rationale. Update the relevant current, decision, reference, or runbook document in the same commit as the code or operational change it describes.
