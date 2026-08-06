# 20M / 100M Validation OOM and Recovery Decision

_Date: 2026-08-06 Europe/Rome_

## Incident

The first 20M-parameter model run on the fixed 100M-token dataset completed optimizer update 500 and then failed during scheduled held-out validation on an NVIDIA T4.

Ordinary training was stable at approximately 4.0k target tokens per second and approximately 9.1 GiB reported VRAM. The failure was deterministic at the evaluation boundary, not a training overflow or model-divergence event.

The evaluation path forwarded the complete 16-sequence validation block at context 2,048 in one call. With a semantic vocabulary of 50,257, this created approximately 3.07 GiB of FP16 logits and then requested approximately 6.14 GiB for cross-entropy work. Those allocation sizes matched the CUDA OOM evidence.

Training did not have the same problem because it already split each 16-sequence block into microbatches of four.

## User decision

The user decided:

- do not preserve or recover the local step-250 checkpoint as a requirement;
- remove the repository-imposed 749-update default stop;
- attempt the complete remaining one-pass 100M-token schedule in one Kaggle invocation;
- run validation, local checkpointing, and verified remote publication every 250 successful optimizer updates;
- keep explicit bounded-session support through `--max-steps-this-session` for diagnostics or manual overrides.

The absence of the artificial 749-update stop does not override Kaggle platform limits or unexpected session termination. The 250-step verified remote publication cadence bounds rollback when the platform interrupts the run.

## Implemented validation fix

Held-out evaluation now:

- runs under `torch.inference_mode()`;
- defaults to a dedicated validation microbatch of one sequence, independent of the training microbatch;
- releases optimizer gradients with `zero_grad(set_to_none=True)` before evaluation;
- clears unused CUDA allocator cache before and after evaluation;
- deletes each microbatch's logits and loss immediately after accumulation;
- restores the model's previous train/eval mode in a `finally` block;
- enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the Kaggle entry point as a secondary fragmentation safeguard.

The primary fix is validation microbatching. Allocator configuration is not relied on to make an unsafe full-block evaluation fit.

At validation microbatch one, the dominant full-vocabulary cross-entropy allocation is reduced from approximately 6.14 GiB to approximately 0.38 GiB.

## Revised operational defaults

```text
training microbatch: 4 sequences
validation microbatch: 1 sequence
local checkpoint cadence: 250 updates
validation cadence: 250 updates
verified remote publication cadence: 250 updates
repository default session cap: none within the finite one-pass plan
W&B run ID: 20m-100m-data-004
allocator safeguard: expandable_segments:True
```

The finite trainer plan remains authoritative. Removing the session cap does not permit wraparound or training beyond the exact one-pass update count.

## Launch contract

Use the unchanged official command after pulling `main`:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_100m.py
```

The pinned launch commit must include the validation microbatch fix. The one-click wrapper applies the revised 250-step cadence and full-run default before invoking the fail-closed experiment launcher.
