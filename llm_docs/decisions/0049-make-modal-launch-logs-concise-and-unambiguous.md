---
status: accepted
date: 2026-08-12
---

# 0049 — Make Modal launch logs concise and unambiguous

## Context and problem statement

A 100M/2B Modal launch was manually stopped after the operator interpreted two kinds of startup output as an error loop: the qualification CLI printed the full `train.block_ids` list containing more than fifteen thousand integers, and PyTorch emitted a red stderr warning because NumPy was not installed in the Modal image. Historical Modal container logs showed no Python traceback or application failure for that attempt; the visible `cpu = _conversion_method_template(...)` line belonged to the missing-NumPy warning.

The excessive qualification output made it difficult to see whether startup was progressing toward checkpoint restore, microbatch qualification, and training, and it obscured genuinely actionable stderr. The full qualification plan is already persisted to the run directory, so duplicating its large arrays to stdout provides no durability or reproducibility value.

## Considered options

- Keep all current output and rely on operators to distinguish warnings from failures manually.
- Remove only the NumPy warning while leaving the full qualification plan on stdout.
- Preserve full plan artifacts on disk, make file-producing qualification calls emit only concise status output, install NumPy in the Modal image, and add explicit launcher stage records.

## Decision outcome

Chosen option: **make Modal startup logs concise while preserving full artifacts and interactive CLI behavior.**

The operational contract is:

- The Modal image explicitly installs NumPy so PyTorch does not emit the `Failed to initialize NumPy` warning during normal startup.
- `python -m dataset.qualification report` continues to print the complete derived plan when no `--output` path is supplied, preserving the existing interactive/inspection CLI behavior.
- When `--output` is supplied, the qualification CLI writes the complete plan to that file and emits only a compact JSON status record containing the profile and output path. Large arrays such as `train.block_ids` are not duplicated to stdout.
- The Modal launcher emits compact JSON `modal_stage` records for dispatch, remote-runtime start, and remote-runtime completion so operators can distinguish launcher progress from trainer logs.
- The full qualification plan remains persisted under the run directory and therefore remains available for reproducibility, debugging, and contract checks.
- These changes are observability/runtime-environment changes only. They do not alter dataset bytes, block ordering, model architecture, optimizer, schedule, precision, microbatch selection policy, checkpoint cadence, checkpoint format, or training trajectory.

## Consequences

### Positive

- Startup logs no longer contain thousands of block-ID lines that can be mistaken for repeated training output or an exception payload.
- The harmless PyTorch NumPy warning is removed from normal Modal stderr.
- Operators get explicit machine-readable stage markers around the remote execution lifecycle.
- Qualification plan reproducibility is unchanged because the complete JSON remains written to disk.
- Interactive qualification inspection still prints the full JSON when no output file is requested.

### Negative or limiting

- File-producing qualification invocations no longer expose the full plan directly on stdout; callers that need the full structure must read the requested output file.
- The launcher stage records cover the orchestration boundary rather than every internal runtime sub-stage; the trainer and runtime command lines remain the detailed source for those transitions.
- Adding NumPy slightly increases the Modal image dependency surface.

## Validation

This decision is satisfied when all of the following hold:

1. A qualification report invoked with `--output` writes the complete plan, including `train.block_ids`, to disk but does not print that array to stdout.
2. A qualification report invoked without `--output` still prints the complete JSON plan.
3. The Modal image contains NumPy and no longer emits the PyTorch `Failed to initialize NumPy` warning under normal startup.
4. Modal launch logs contain `modal_stage` records for dispatch, remote start, and remote completion.
5. Focused regression tests cover the qualification stdout behavior.

## Links

- [`../../modal/launch.py`](../../modal/launch.py)
- [`../../dataset/qualification.py`](../../dataset/qualification.py)
- [`../../tests/test_qualification_output.py`](../../tests/test_qualification_output.py)
- [`0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md`](0047-use-hf-storage-bucket-for-modal-cross-workspace-checkpoints.md)
- [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
