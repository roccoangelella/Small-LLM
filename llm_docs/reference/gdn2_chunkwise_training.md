# GDN-2 reference recurrence and chunkwise training

_Last reviewed: 2026-08-13_

## Scope

The repository retains readable PyTorch GDN-2 implementations as the mathematical/correctness reference. **They are not the selected production CUDA training backend.** Production CUDA execution is the qualified `fla-core==0.5.2` adapter described in [`gdn2_fla_backend.md`](gdn2_fla_backend.md).

Available reference forms include tokenwise recurrence, direct WY-style chunkwise execution, and adaptive subchunking for numerical diagnostics/fallback. Backend implementations must preserve the same recurrence, learned parameters, state-dict keys, and checkpoint semantics.

## Mathematical recurrence

For each head, GDN-2 updates matrix state `S_t` with decay, erase, and write controls:

```text
e_t = b_t * k_t
z_t = w_t * v_t
S_bar_t = Diag(exp(g_t)) S_(t-1)
S_t = S_bar_t + k_t (z_t - S_bar_t^T e_t)^T
o_t = S_t^T q_t / sqrt(d_k)
```

The tokenwise oracle evaluates this directly. The PyTorch chunkwise reference groups tokens, factors cumulative decay, solves the lower-triangular WY system, and uses dense intra-chunk products while carrying recurrent state between chunks.

## Precision/correctness boundary

Reference recurrence/state and numerically sensitive decay/chunk auxiliaries execute in FP32 for correctness comparisons. Optimized mixed-precision execution is checked against this boundary under the real trainer AMP contract.

No backend is allowed to silently clip learned decay, skip non-finite regions, change token order, or mutate model/checkpoint geometry to make a kernel pass.

## Serialized versus runtime chunk

Completed production checkpoints serialize:

```text
gdn_chunk_size = 32
```

The selected FLA CUDA backend executes an internal runtime chunk of 64. That internal choice is an execution detail, not a reason to relabel checkpoint geometry. PyTorch reference/adaptive tools may use diagnostic chunking as needed without changing the learned model.

## Verification contract

Reference and optimized paths are qualified by output, final-state, and gradient agreement on deterministic synthetic and real-checkpoint cases, plus finite-gradient checks. The accepted production qualification used a corrected FP32 adaptive oracle and the real step-4000 next-block forward/backward gate; see [`gdn2_fla_backend.md`](gdn2_fla_backend.md) and the linked evidence there.
