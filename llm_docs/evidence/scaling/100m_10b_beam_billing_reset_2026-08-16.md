# 100M / 10B Beam billing-baseline reset

_Observed 2026-08-16 UTC._

The previous monitor estimate used the cumulative W&B `_runtime` from before
the Beam account/workspace change. That estimate was retired rather than
carried into the new account's budget.

At `2026-08-16T15:13:23.755281Z`, the hourly supervisor wrote a billing reset
marker with W&B runtime `79315.065993758` seconds as the baseline. Hugging Face
latest was still `step-00015500`; the W&B run was `100m-10b-data-001`. The
monitor then reported:

- account-cost basis: `$0.00` (`account_zero` mode);
- notional cap basis: `$0.00` since reset;
- notional resource rate: `$2.10/hour` for RTX4090 + 4 CPU cores + 32 GiB RAM;
- budget: `$30.00`.

The dry-run selected a 7,733-step bounded segment from `step-00015500`. The
segment was launched at `2026-08-16T15:13:40Z`; its first Beam task was
`adfaa8a7-99ac-490a-a0b7-1d6165b027e6` in workspace `e77bbf`. The hourly cron
entry remains installed in UTC with `SMALL_LLM_BEAM_BILLING_MODE=account_zero`.

The reset marker is local operational state at
`/tmp/small-llm-beam-monitor/billing_reset.json`. The supervisor refuses to
allocate if that marker is missing, so an old cumulative W&B runtime cannot
silently consume the new account budget.
