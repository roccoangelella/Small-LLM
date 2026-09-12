---
status: proposed
date: 2026-09-12
supersedes: null
---

# 0180 — Opt-in torch.compile lane for the MoE decoder blocks

## Context and problem statement

A profiled update of the accepted production geometry (ADR-0173/0174/0177: 64 experts, Top-2,
`d_model=256`, 8 layers, expert width 352, 8,000/8,192 vocabulary, GDN-2 hybrid via `fla-core==0.5.2`
with three GDN blocks per MHA block, BF16 autocast, batched dropless expert GEMM in
`MOE_model/model.py::DroplessTopKMoE`) spends roughly **60 % of GPU time in unfused elementwise and
copy kernels** — at microbatch 8, about 4,398 `aten::mul`, 6,517 `copy_`, ~6,000 `add` and ~4,700
bf16 cast launches per update — against roughly 30 % GEMM and 10 % FLA GDN kernels. The architecture
is frozen, so the remaining lever is execution: fusing those elementwise and copy chains.

`torch.compile` (Inductor) is the candidate. Three properties of this model make it non-obvious:

1. The expert dispatch reads `capacity = int(counts.max())` once per microbatch — a deliberate host
   synchronisation whose value changes with routing imbalance. A naive compile either refuses to
   trace it or recompiles on every new capacity, which would cost more than it saves.
2. The GDN-2 mixer calls FLA's `ChunkGDN2Function`, a custom autograd Function wrapping Triton
   kernels. It is qualified numerics (ADR-0018/0020/0021) and must not be rewritten.
3. Quantile Balancing (ADR-0174) keeps host-side Python state — a per-step target rank and a score
   frontier carried across gradient-accumulation microbatches — that decides the selection bias.
   Any tracing of that bookkeeping risks silently changing the bias, i.e. the model.

A fourth constraint is contractual: the checkpoint identity is `MoEModelConfig`, and a launch that
compiles must stay loadable by a launch that does not.

## Considered options

- **A — Do nothing.** Keep eager execution and accept the elementwise overhead.
- **B — Compile the whole model** (`torch.compile(model)`), one region per forward.
- **C — Compile each `MoEDecoderBlock` in place, opt-in behind a trainer-side `--compile` flag**,
  with the Quantile Balancing bookkeeping explicitly excluded from tracing.
- **D — Hand-fuse the hot elementwise chains** (custom Triton or fused ops in `DroplessTopKMoE`).
- **E — Put the compile mode in `MoEModelConfig`** so a checkpoint records how it was executed.

## Decision outcome

Chosen option: **C**, exposed as `--compile {off,blocks}` (default `off`) on the MoE entrypoint and
applied by `MOE_model/compile_lane.py::apply_compile_lane(model, mode)` from `MOE_model/setup.py`,
after the model is built and before the engine builds the optimizer.

Because:

- **Backward matches eager autocast scope.** Forward in `MOE_model/step.py` runs
  under autocast; `.backward()` runs after leaving that context. When the lane is
  armed, it sets `torch._functorch.config.backward_pass_autocast = "off"` if the
  setting exists (`hasattr` guard), overriding PyTorch 2.10's `"same_as_forward"`
  default. The setting stays active for lazy tracing and later recompilation;
  it is process-global. With the lane off, it is unchanged.
- **Identity is untouched.** `nn.Module.compile()` installs a compiled `_compiled_call_impl` on the
  block itself instead of wrapping it in an `OptimizedModule` child, so `state_dict()` keys, the
  parameter objects and their `id()`s are unchanged; the optimizer built afterwards captures exactly
  the same tensors and checkpoints load in both directions. Option E is rejected for the same
  reason the ADR-0177 contract keeps launch settings out of the model config: compilation is
  execution, not architecture, and recording it would fork the checkpoint identity for free.
- **The block is the right region.** It spans the two norms, the mixer, router scoring, the padded
  expert SwiGLU GEMM and the recombination — that is where the profiled `mul`/`copy_`/`add`/cast
  traffic lives — while keeping the mixer's unavoidable breaks local to one block. Option B buys
  nothing extra, because the first break inside a block splits the region anyway.
- **The varying capacity is handled by `dynamic=True`, not by a config override.** Dynamo cannot
  trace `int(counts.max())` with the default `capture_scalar_outputs=False`, so it breaks the graph
  there; the padded buffer, the three `bmm` calls and the recombination land in the following graph
  with `capacity` as a plain Python integer, and `dynamic=True` makes that integer a dynamic symbol
  from the first compile. Measured: capacities 18/23/18/38 on one MHA block, and 15/16/19 across a
  whole model, reuse a single compilation with no recompile. Setting
  `capture_scalar_outputs=True` was tried and **rejected**: it makes capacity an *unbacked* symbol
  and Inductor then fails to lower the padded `torch.bmm` with
  `GuardOnDataDependentSymNode: Could not guard on ... Eq(u0, 1)`. **The lane therefore sets no
  `torch._dynamo.config` value at all.**
- **The FLA kernel keeps working by breaking, not by being rewritten.** Option D is rejected for now
  for the same reason: it is a much larger, less reversible change, and it should be justified by
  what Inductor leaves on the table, not assumed before measuring.
- **Quantile Balancing is excluded, not traced.** `SwitchTopKRouter.begin_step`, `_observe_scores`
  and `commit_load` are wrapped once with `torch.compiler.disable` when the lane is armed;
  `_observe_scores` is the only one called from inside a compiled block, and its exclusion is what
  splits the router graph. Measured on CPU with `aot_eager`: the selection bias after one full logical step
  is **bit-identical only when the router inputs are identical**. Keeping the
  bookkeeping eager does not guarantee identical upstream scores on GPU.

Option A remains the status quo and stays in force until the GPU measurement below lands: this ADR
is `proposed`, not `accepted`, and the default is unchanged.

## Consequences

### Positive

- A production command adds `--compile blocks` and changes nothing else; every existing launch,
  checkpoint and resume path is byte-for-byte unaffected with the flag absent.
- The whole elementwise/copy region after the capacity sync sits in one graph, which is exactly the
  region the profile blames.
- The lane is falsifiable: the graph-break count, the absence of capacity recompiles, numeric
  agreement and bias bit-identity for identical router inputs are asserted on CPU in `tests/test_moe_compile_lane.py`.

### Negative or limiting

- **Rounding changes.** Inductor refuses no reassociation the eager kernels make, but it does fuse
  and reorder them, so compiled BF16 training is not bit-identical to eager training. Adoption
  therefore needs a learning comparison, not only a throughput number. The H100
  Inductor measurement already shows a rounding-level difference in first-update
  `layer0_bias_absmax`: **0.0332623720 eager vs 0.0332735777 compiled**. This
  disproves unconditional bias bit-identity; it does not establish harmful drift.
- Compilation is not free: one warm-up cost per distinct graph at the start of a run, and Inductor
  autotuning time on top if it is ever enabled.
- Graph breaks remain. Measured on CPU with `aot_eager`: an MHA block is **8 graphs / 7 breaks**, a
  GDN-2 block **19 graphs / 18 breaks**. Break reasons: the deliberate `torch.compiler.disable` on
  `_observe_scores`, `Tensor.item()` (the capacity sync), `bincount`'s data-dependent output shape,
  and — in the GDN block only — the CPU adaptive chunkwise fallback's data-dependent bisection loop.
  **The GDN number is CPU-specific**: on CUDA that whole fallback is replaced by FLA's
  `ChunkGDN2Function`, which has not been counted here because this machine has no GPU.
- `torch.compiler.disable` is applied to `SwitchTopKRouter` at class level, so once the lane is armed
  it is in force for every router in that process. It is semantically transparent (it only tells
  Dynamo to stop tracing) but it is process-global state.
- The CPU-only adaptive GDN fallback re-specialises when the microbatch sequence count changes
  (28 → 37 frames going from 2 to 3 sequences). The router and expert dispatch do not.
- The lane buys nothing on the **version-2 grouped dispatch** (`DroplessTop1MoE`, and
  `DroplessTopKMoE` with `batched=False`). Those slice one Python loop iteration per expert, so
  Dynamo re-specialises on every distinct expert-group size and reaches `recompile_limit` before
  falling back to eager for that frame — observed on an end-to-end `--model-size smoke
  --compile blocks` launch, which still completed correctly (exit 0, checkpoint written, Inductor
  backend, CPU). The lane does not refuse it, because it is a warm-up cost and not an error, but
  `--compile blocks` is only meaningful with `dropless_padded_batched_gemm`, i.e. the accepted
  geometry.

## Validation

Already measured (CPU, `tests/test_moe_compile_lane.py`, `backend="aot_eager"`):

- compiled versus eager loss and every parameter gradient agree within FP32 tolerance on identical
  weights and inputs;
- an MHA block runs backward across two observed expert capacities, with input and
  parameter gradients matching eager within FP32 tolerance; an execution counter
  asserts that compiled frames actually run for each capacity;
- `state_dict()` keys, `named_parameters()` and parameter `id()`s are unchanged with the lane armed,
  and checkpoints cross-load between an eager and a compiled model;
- the Quantile Balancing selection bias after one logical step is bit-identical
  only with identical router inputs in the CPU test;
- graph counts are recorded in the test log and asserted bounded (MHA ≤ 10, GDN ≤ 24);
- three consecutive microbatches with different expert loads, and four microbatch sizes on one
  block, reuse one compilation with zero recompiles;
- an end-to-end `python -m MOE_model ... --compile blocks` launch completes one optimizer update and
  writes its checkpoint on CPU with the default Inductor backend.

Still required before this ADR can move to `accepted` — **on the production GPU, accepted geometry,
BF16 autocast, Inductor backend**:

1. **A measured ≥ 20 % gain** in tokens/second (or equivalently ms/update) at the production
   microbatch, against the same commit with `--compile off`, warm-up excluded, plus a re-profile
   showing the elementwise/copy share actually falling from ~60 %.
2. **A learning comparison**, because the rounding changes: identical seed, identical data order,
   loss curves compared over enough updates to separate a real divergence from step noise.
3. Confirmation that the FLA GDN-2 path still executes the qualified kernel under compilation, and
   the graph/break count for a GDN block on CUDA.

If (1) misses 20 % or (2) shows divergence, the lane stays `off` and this ADR is rejected rather
than weakened; the fallback lever is then option D, hand-fusing the dispatch.

## Links

- `llm_docs/decisions/0173-wire-the-accepted-64-expert-top2-geometry-with-batched-expert-gemm.md`
- `llm_docs/decisions/0174-adopt-the-owner-accepted-moe-router-numerical-contract.md`
- `llm_docs/decisions/0177-accepted-moe-production-launch-contract.md`
- `llm_docs/decisions/0021-qualify-fla-gdn2-v052-and-resume-step4000.md`
- `MOE_model/compile_lane.py`, `MOE_model/setup.py`, `MOE_model/__main__.py`
- `tests/test_moe_compile_lane.py`
