# 100M / 10B Beam monitor shutdown

_Observed 2026-08-17 UTC._

The live VPS still had the cron entry
`small-llm-100m-10b-budget-monitor` despite an earlier check reporting no
entry. The live entry ran at minute 17 UTC. After the previous training task
failed, it relaunched a bounded 3,769-step segment through local launcher PID
`1131603`; Beam GPU task
`169b7e0e-910d-4076-b1ab-7cdbb15719ce` entered `RUNNING` at
`2026-08-17T06:18:53Z`.

The exact Beam task was stopped, the exact Small-LLM cron line was removed,
and the local launcher was terminated. A follow-up live check found no
matching cron entry, no local `monitor_100m_10b_beam` or `beam/vps_train.py`
process, and no active Beam task.
