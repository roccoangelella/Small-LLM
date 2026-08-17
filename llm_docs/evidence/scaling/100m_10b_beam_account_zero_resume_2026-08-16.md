# 100M / 10B Beam account-zero resume

_Observed 2026-08-16 UTC._

The Beam account/workspace was changed after the earlier RTX4090 segment. The
existing `.env` values were installed as Beam secrets under the canonical
names `WANDB_API_KEY`, `HF_TOKEN`, and `SMALL_LLM_HF_REPO_ID`. Hugging Face
latest remained the verified `step-00015500` checkpoint, with source commit
`1f9dff920ecc45ce2fdb43fd875514a18391273d` and microbatch 4.

The new Beam workspace (`e77bbf`) initially had fresh `small-llm-runs`,
`small-llm-cache`, and `small-llm-data` volumes. CPU preflight and dataset
staging succeeded, but the first GPU task
`c467707d-dd15-4b3c-b82c-9063792623a1` failed before trainer startup because
the new run volume lacked `modal_runtime.json`. A one-time CPU helper seeded
that runtime contract from the verified checkpoint identity; it did not alter
model, dataset, optimizer, schedule, precision, or checkpoint bytes.

The exact resume was relaunched through the hourly supervisor. Task
`d868a9b8-4e12-45c8-b287-ad6a7ea7988c` entered `RUNNING` at
`2026-08-16T14:56:21Z` on `beam.vps_train:train_vps_rtx4090_remote` with the
trainer command resuming `step-00015500`, 60,794 remaining updates, and
microbatch 4. W&B returned to `running`; at the last observation the trainer
was still in startup/restore and HF latest had not advanced beyond step 15,500.

The local hourly supervisor runs in `account_zero` mode per the user's Beam
account change, but the hard cap is deliberately based on the notional
serverless estimate. At `2026-08-16T15:08:13Z`, the estimate reached `$51.28`
including the safety margin, so the supervisor stopped task
`d868a9b8-4e12-45c8-b287-ad6a7ea7988c`. A follow-up check found no running Beam
task or container and refused relaunch with `budget_exhausted`. The account's
reported `$0` actual charge remains visible separately; it does not bypass the
requested notional `$30` resource cap. The supervisor remains fail-closed if
Beam, W&B, or HF control-plane reads fail.
