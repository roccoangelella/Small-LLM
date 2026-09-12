---
status: accepted
date: 2026-09-06
supersedes: null
---

# ADR 0155 — Freeze Top-1 Switch-style router without load balancing

Date: 2026-09-06
Status: Accepted for `moe-8e-top1` experiment branch
Branch: `moe-8e-top1`

## Context and problem statement

The first Small-LLM MoE experiment converts every one of the 20 dense FFN positions into an independent 8-expert MoE layer. Each expert keeps the current dense SwiGLU geometry (`d_model=512`, `d_ff=1408`). The experiment is Top-1 only and starts from scratch.

The previous planning discussion considered explicit load-balancing mechanisms. The user rejected imposing an externally preferred equal-load target for the first experiment. The initial goal is to observe whether learned routing naturally produces useful specialization and how utilization distributes across experts, rather than forcing a target distribution.

## Considered options

No alternative was recorded at the time; section added for the template.

## Decision outcome

For the first MoE experiment:

- Replace the FFN in **all 20 decoder layers** with an independent 8-expert MoE layer (160 independent SwiGLU experts total).
- Use **Top-1 routing only**.
- Use a learned linear router from the normalized FFN input hidden state to 8 router logits.
- Compute router logits and softmax in **FP32** for numerical stability.
- Select the expert with the highest router probability.
- Use **Switch-style combine weighting**: the selected expert output is multiplied by the selected expert's original softmax probability; the selected probability is not renormalized to 1.
- Do **not** use a load-balancing auxiliary loss.
- Do **not** use a loss-free balancing bias or any other hand-controlled expert-utilization target.
- Use **dropless routing**: every token is processed by its selected expert regardless of load imbalance.
- Use router z-loss with initial coefficient **lambda_z = 1e-4**.
- Route the router matrix through **AdamW** with the same base learning rate as the run (`lr_scale=1.0`).
- Route expert SwiGLU gate/up/down matrices through the project's existing **Muon** role, matching the dense FFN optimizer treatment.
- Train the MoE model **from scratch**; expert weights are independently initialized using the same initialization family as the current dense FFN rather than cloned from a trained checkpoint.
- No Top-2 implementation is part of this branch's initial scope.

## Telemetry requirement

Because there is no balancing objective, routing behavior is a first-class experimental result. W&B must record per-layer and aggregate expert utilization, including at least token counts/fractions per expert, routing entropy/confidence statistics, dead/near-dead experts, max/min utilization, coefficient of variation or equivalent imbalance statistic, z-loss, router-logit magnitude statistics, and global throughput/step-time/memory metrics.

## Checkpoint identity

The checkpoint must fail closed on the complete MoE architecture and routing contract. It must serialize or identify all MoE-critical fields, including expert count, Top-k, expert width/type, MoE layer placement, router type/version, router scoring/selection/combine rules, router precision, z-loss coefficient, dropless policy, dispatch implementation identity, router optimizer role and LR scale, expert optimizer roles, total/active parameter accounting, and the normal model/trainer/data/RNG/checkpoint state required for exact resume.

## Rationale

This deliberately separates two questions:

1. Can sparse learned Top-1 routing discover useful expert specialization without an externally imposed utilization target?
2. If routing collapses or becomes pathologically imbalanced, does the observed evidence justify adding a balancing mechanism in a later controlled ablation?

The router therefore follows the useful Top-1 property of Switch routing—retaining the selected softmax gate as a differentiable combine weight—without inheriting Switch's auxiliary load-balancing objective.

## Consequences

- Uneven or collapsed expert usage is allowed and must be measured rather than corrected automatically.
- Dropless execution preserves every token's computation but can reduce throughput when routing is highly imbalanced.
- Router z-loss regularizes router-logit scale but is not a load-balancing mechanism.
- Any later balancing mechanism or Top-2 routing requires a separate decision and experiment identity.
