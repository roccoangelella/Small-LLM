# PyTorch upstream review for the project's GDN-2 implementation

_Date: 2026-08-07_

## Executive assessment

The project has a real independent PyTorch implementation of the Gated Delta Rule-2 recurrence, a differentiable WY-style chunkwise formulation, a recurrent correctness oracle, cache semantics, gradient-parity tests, and a numerical fallback that successfully carried the 20M model through a complete 100M-token pretraining run.

That is enough to justify an upstream discussion with PyTorch, but **not enough to send `torch.nn.GatedDeltaNet2` as a direct PR yet**.

The best first upstream proposal is a reusable **Gated Delta Rule-2 functional/reference primitive** (or an RFC asking maintainers where such a primitive belongs), rather than the Small-LLM-specific full token-mixer module. PyTorch's current documentation explicitly encourages fast-moving transformer-like architectures to be assembled from core building blocks or ecosystem libraries, and its contribution guide asks contributors to discuss new core features in an issue before sending a PR.

## Licensing and provenance

The Small-LLM repository is MIT licensed.

The official NVIDIA Gated DeltaNet-2 repository is under the NVIDIA Source Code License-NC. Therefore no PyTorch contribution should copy, port, translate, or mechanically derive from NVIDIA's source code.

Any upstream contribution from this project should document provenance clearly:

- recurrence and chunkwise equations derived from the public paper;
- implementation authored independently in Small-LLM;
- no source copied from the NVIDIA NC implementation;
- commit history retained as evidence of the project's implementation path;
- paper and authors cited for the algorithm.

Before upstreaming, perform a human provenance audit of the commits that introduced `model/gdn2.py` and later stability changes.

## What is strong already

### Mathematical recurrence

The recurrent oracle implements

```text
S_bar = D_t S_(t-1)
e_t = b_t * k_t
z_t = w_t * v_t
S_t = S_bar + k_t (z_t - S_bar^T e_t)^T
```

which is algebraically the Gated Delta Rule-2 update in the paper.

### Chunkwise structure

The chunkwise path contains the expected WY-style ingredients:

- cumulative channel-wise log decay;
- asymmetric decay-normalized erase/key factors;
- strictly lower intra-chunk matrix;
- unit-lower-triangular solve;
- erase/write auxiliaries;
- inter-chunk FP32 recurrent state;
- dense causal output construction.

### Independent oracle and gradient checks

`tests/test_gdn2.py` compares chunkwise outputs and final states against the tokenwise recurrent oracle and optionally compares gradients for Q, K, V, log-decay, erase gate, write gate, and initial state.

It also tests causality and one-shot versus tokenwise/segmented cache execution.

`tests/test_gdn2_stable.py` additionally tests a strong-decay case with gradient parity and confirms the assembled model uses the adaptive backend.

### Real training evidence

The implementation is not only a toy unit test. The 20.6M-parameter hybrid completed its finite 100M-token run on a T4, including numerical recovery, exact resume/replay evidence, and final held-out validation loss 4.252758 / perplexity 70.299.

## Major blockers before a PyTorch-core PR

### 1. The current full module is project-specific

`GatedDeltaNet2` takes the project's `ModelConfig` and custom `RMSNorm`, fixes several architectural choices, and packages projections, convolution, gates, recurrence, output normalization, and output projection together.

For PyTorch core, the more reusable object is the recurrence/chunkwise primitive. A high-level paper-faithful layer can be built on top once maintainers agree it belongs in core.

### 2. Missing generality relative to the paper

The current module rejects unequal key/value head counts and unequal key/value dimensions. The paper explicitly supports grouped value heads (`H_v > H`) and distinct key/value axes.

A core primitive should support at least:

```text
q, k: [B, T, H_k, D_k]
v, write: [B, T, H_v, D_v]
erase, log_decay: key-head/key-channel side
state: grouped/repeated mapping defined explicitly
```

or explicitly scope v1 to equal heads only with maintainer agreement.

### 3. No packed variable-length support

The paper's production implementation resets recurrent state at packed sequence boundaries. The current Small-LLM implementation handles dense batches only.

For a core primitive, variable-length semantics should be designed up front, even if the first accepted implementation only supports dense sequences.

### 4. `torch.compile` / export blocker

The current reference path contains tensor-to-Python control flow such as:

```python
bool(torch.isfinite(...).all())
```

and the adaptive path additionally calls `.item()` / `float(...)` on GPU-derived span values and uses data-dependent Python loops and slicing.

A local PyTorch 2.10 full-graph compile probe fails on the tensor `bool` conversion. This should be treated as an upstream blocker for a modern core primitive.

The eager reference can retain explicit validation outside compiled regions, but the hot numerical path should be tensor/dispatcher friendly.

### 5. Adaptive backend is correctness-first, not an upstream performance design

`AdaptiveChunkwiseGDN2Backend` computes a decay span for every proposed chunk, transfers the scalar decision to Python, and may repeatedly recompute cumsums while bisecting 32 -> 16 -> 8 -> 4 -> 2 -> 1.

The span reduction takes the maximum across batch, heads, and key channels, so one extreme coordinate forces the entire current batch chunk to shrink.

With six GDN layers at context 2048, this can create hundreds to thousands of device synchronizations and small sequential chunk calls per forward when learned decay becomes extreme.

This is strongly consistent with the observed 100M-run compute slowdown (~3830 target tok/s early to ~445 tok/s late), although the run did not log actual subchunk sizes, so causal attribution remains unproven.

Do not upstream the adaptive Python wrapper as the final performance path.

### 6. Dtype semantics need redesign

The recurrence and chunkwise functions currently call `.float()` unconditionally. This correctly protects FP16/BF16 training state, but also downcasts FP64 inputs to FP32 internally. A core reference should preserve true FP64 arithmetic for numerical verification while using FP32 accumulation for lower-precision model dtypes.

The paper itself reports FP64 reference checks to machine precision, making this an important gap.

### 7. Initialization/API fidelity

The paper specifies Xavier-uniform initialization with gain `2^-2.5` for its linear layers and zero biases. The Small-LLM full layer mostly inherits PyTorch default initializers.

That is acceptable for the project's experiment if intentionally chosen, but an upstream class named `GatedDeltaNet2` should either follow the paper's reference parameterization or clearly expose/document a different initialization contract.

The constructor also needs normal PyTorch factory kwargs (`device`, `dtype`) and should avoid project-specific config objects.

### 8. PyTorch integration expectations

Before core submission, test/support expectations should include:

- CPU reference execution;
- CUDA execution;
- FP64 oracle and `gradcheck`/where practical `gradgradcheck`;
- FP32, BF16, FP16 tolerance matrix;
- autocast;
- non-contiguous inputs;
- device/dtype factory kwargs and meta-device construction where applicable;
- serialization/state_dict;
- `torch.compile` / AOTAutograd behavior;
- export behavior or explicit unsupported contract;
- deterministic one-shot vs segmented/recurrent cache semantics;
- variable sequence lengths / boundary reset semantics;
- broader random geometry sweep (sequence lengths, H, D_k, D_v, initial state);
- numerical stress cases near decay limits.

## Recommended upstream API direction

Do not start by proposing:

```python
nn.GatedDeltaNet2(ModelConfig(...))
```

Instead, propose an RFC around a generic primitive, for example conceptually:

```python
gated_delta_rule(
    q,
    k,
    v,
    log_decay,
    erase_gate,
    write_gate,
    initial_state=None,
    *,
    chunk_size=64,
    return_final_state=True,
)
```

The exact namespace should be decided with maintainers. Candidates to discuss include a functional API or an attention/recurrent building-block namespace. Do not freeze the public name before maintainer feedback.

The primitive is a stronger core pitch because tying gates recovers related delta-rule mechanisms, making it more reusable than a single model wrapper.

A later paper-faithful module could then provide projections, short convolution, normalization, gate parameterization, and caching on top of the primitive if maintainers want it in core. Otherwise that high-level module can live in an ecosystem package.

## Recommended contribution sequence

1. **Do not open a code PR first.**
2. Open a PyTorch feature/RFC issue describing the reusable primitive, research adoption, paper, independent implementation, real training evidence, and licensing/provenance situation.
3. Ask maintainers explicitly whether the desired home is core `torch.nn`, a lower-level functional primitive, `torch.nn.attention`, or ecosystem.
4. In Small-LLM, refactor the recurrence into a standalone dependency-free reference file and close the blockers above.
5. Add an upstream-style test matrix and compile/export probes.
6. Benchmark eager reference and any optimized path separately.
7. Only after positive maintainer direction, create a minimal PyTorch branch/PR touching the smallest possible surface.
8. Treat a fused Triton/CUDA backend as a separate follow-up unless maintainers request it in the initial scope.

## Acceptance likelihood

### Direct `torch.nn.GatedDeltaNet2` PR now

**Low.** The paper is recent, the full module is architecture-specific, PyTorch core has a high bar for new APIs, and the current implementation has compile, generality, dtype, and performance gaps.

### RFC for a reusable Gated Delta Rule-2 primitive

**Worth doing.** It has a stronger core-building-block argument, independent reference code, a mathematical oracle, real training evidence, and a licensing reason why a clean permissive reference implementation could be useful.

### Ecosystem/reference package if core declines

**High-value fallback.** The implementation could still be published as a clean MIT reference package with a paper-faithful API, correctness oracle, tests, and later optimized backend. This may also strengthen a future PyTorch-core case by demonstrating external adoption.

## Bottom line

The project has something real enough to bring to PyTorch maintainers, but the right next action is an RFC, not a surprise `torch.nn.GatedDeltaNet2` PR. The upstreamable asset is the independently implemented Gated Delta Rule-2 primitive and its correctness harness. The project-specific full layer and adaptive Python fallback need further refactoring before they should be proposed as core API.
