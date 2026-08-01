"""Readable recurrent and differentiable chunkwise PyTorch GDN-2 backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .components import RMSNorm
from .config import ModelConfig


@dataclass
class GDN2Cache:
    """GDN-2 cache: state ``[B,H,K,V]`` and raw history ``[B,k-1,D]``."""

    recurrent_state: Tensor
    q_history: Tensor
    k_history: Tensor
    v_history: Tensor


def _check_recurrence_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_decay: Tensor,
    erase_gate: Tensor,
    write_gate: Tensor,
    initial_state: Tensor | None,
) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,K], got {tuple(q.shape)}")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have shape [B,T,H,K/V]")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have matching shapes, got {q.shape} and {k.shape}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q in batch, sequence, and head dimensions")
    for name, tensor, expected_tensor in (
        ("log_decay", log_decay, q),
        ("erase_gate", erase_gate, q),
        ("write_gate", write_gate, v),
    ):
        if tensor.shape != expected_tensor.shape:
            expected = tuple(expected_tensor.shape)
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None and initial_state.shape != (batch, heads, key_dim, value_dim):
        raise ValueError(
            "initial_state must have shape "
            f"[{batch},{heads},{key_dim},{value_dim}], got {tuple(initial_state.shape)}"
        )
    return batch, sequence, heads, value_dim


def gdn2_recurrent_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_decay: Tensor,
    erase_gate: Tensor,
    write_gate: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the tokenwise oracle on ``[B,T,H,*]`` tensors."""

    _, _, _, value_dim = _check_recurrence_inputs(
        q, k, v, log_decay, erase_gate, write_gate, initial_state
    )
    state = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(
            q.shape[0], q.shape[2], q.shape[3], value_dim, device=q.device, dtype=torch.float32
        )
    )
    outputs: list[Tensor] = []
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    log_decay_float = log_decay.float()
    erase_float = erase_gate.float()
    write_float = write_gate.float()
    for index in range(q.shape[1]):
        state_bar = torch.exp(log_decay_float[:, index]).unsqueeze(-1) * state
        e = erase_float[:, index] * k_float[:, index]
        z = write_float[:, index] * v_float[:, index]
        residual = torch.einsum("bhkv,bhk->bhv", state_bar, e)
        state = state_bar + k_float[:, index].unsqueeze(-1) * (z - residual).unsqueeze(-2)
        outputs.append(
            torch.einsum("bhkv,bhk->bhv", state, q_float[:, index]) / math.sqrt(q.shape[-1])
        )
    if outputs:
        output = torch.stack(outputs, dim=1).to(dtype=q.dtype)
    else:
        output = q.new_empty((q.shape[0], 0, q.shape[2], value_dim))
    return output, state


def gdn2_chunkwise_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_decay: Tensor,
    erase_gate: Tensor,
    write_gate: Tensor,
    initial_state: Tensor | None = None,
    *,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """Run a differentiable WY-style chunkwise GDN-2 training path.

    The sequence is recurrent only across chunks. Inside each chunk, the
    decay-normalized asymmetric delta recurrence is evaluated with cumulative
    sums, a small unit-lower-triangular solve, and dense matrix products. All
    arithmetic that advances the state or forms cumulative decay is FP32; the
    token output is cast back to the query dtype.
    """

    batch, sequence, heads, value_dim = _check_recurrence_inputs(
        q, k, v, log_decay, erase_gate, write_gate, initial_state
    )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    key_dim = q.shape[-1]
    state = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(batch, heads, key_dim, value_dim, device=q.device, dtype=torch.float32)
    )
    if sequence == 0:
        return q.new_empty((batch, 0, heads, value_dim)), state

    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    log_decay_float = log_decay.float()
    erase_float = erase_gate.float()
    write_float = write_gate.float()
    scale = 1.0 / math.sqrt(key_dim)
    outputs: list[Tensor] = []

    for start in range(0, sequence, chunk_size):
        end = min(start + chunk_size, sequence)
        current_size = end - start

        q_chunk = q_float[:, start:end].transpose(1, 2)
        k_chunk = k_float[:, start:end].transpose(1, 2)
        e_chunk = (erase_float[:, start:end] * k_float[:, start:end]).transpose(1, 2)
        z_chunk = (write_float[:, start:end] * v_float[:, start:end]).transpose(1, 2)
        cumulative_log_decay = torch.cumsum(
            log_decay_float[:, start:end].transpose(1, 2), dim=2
        )

        # Factor exp(G_r - G_s) around a per-channel midpoint. This avoids
        # explicitly materializing exp(-G) and reduces overflow risk while
        # preserving the exact pairwise decay ratios.
        decay_max = cumulative_log_decay.amax(dim=2, keepdim=True)
        decay_min = cumulative_log_decay.amin(dim=2, keepdim=True)
        decay_center = 0.5 * (decay_max + decay_min)
        left_decay = torch.exp(cumulative_log_decay - decay_center)
        right_decay = torch.exp(decay_center - cumulative_log_decay)
        centered_erase = left_decay * e_chunk
        centered_key = right_decay * k_chunk

        strictly_lower = torch.tril(
            centered_erase @ centered_key.transpose(-1, -2), diagonal=-1
        )
        identity = torch.eye(
            current_size, device=q.device, dtype=torch.float32
        ).view(1, 1, current_size, current_size).expand(
            batch, heads, current_size, current_size
        )
        wy_inverse = torch.linalg.solve_triangular(
            identity + strictly_lower,
            identity,
            upper=False,
            unitriangular=True,
        )

        center_scale = torch.exp(decay_center)
        erase_aux = (wy_inverse @ centered_erase) * center_scale
        write_aux = wy_inverse @ z_chunk
        correction = write_aux - erase_aux @ state

        centered_query = left_decay * q_chunk * scale
        decayed_query = centered_query * center_scale
        causal_qk = torch.tril(centered_query @ centered_key.transpose(-1, -2))
        chunk_output = decayed_query @ state + causal_qk @ correction
        outputs.append(chunk_output.transpose(1, 2))

        final_log_decay = cumulative_log_decay[:, :, -1:, :]
        tail_key = torch.exp(final_log_decay - cumulative_log_decay) * k_chunk
        state = (
            torch.exp(final_log_decay.squeeze(2)).unsqueeze(-1) * state
            + tail_key.transpose(-1, -2) @ correction
        )

    output = torch.cat(outputs, dim=1).to(dtype=q.dtype)
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(state).all()):
        raise ValueError(
            "chunkwise GDN-2 produced non-finite values; reduce gdn_chunk_size "
            "or use a qualified fused backend"
        )
    return output, state


@runtime_checkable
class GDN2Backend(Protocol):
    """Callable backend for the ``[B,T,H,*]`` GDN-2 recurrence."""

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        log_decay: Tensor,
        erase_gate: Tensor,
        write_gate: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...


class PyTorchGDN2Backend:
    """Tokenwise PyTorch oracle used for correctness and recurrent decoding."""

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        log_decay: Tensor,
        erase_gate: Tensor,
        write_gate: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return gdn2_recurrent_reference(
            q, k, v, log_decay, erase_gate, write_gate, initial_state
        )


class PyTorchChunkwiseGDN2Backend:
    """Autograd-compatible PyTorch chunkwise backend for training."""

    def __init__(self, chunk_size: int = 64) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        self.chunk_size = chunk_size

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        log_decay: Tensor,
        erase_gate: Tensor,
        write_gate: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return gdn2_chunkwise_reference(
            q,
            k,
            v,
            log_decay,
            erase_gate,
            write_gate,
            initial_state,
            chunk_size=self.chunk_size,
        )


def _validate_backend_result(
    result: object,
    q: Tensor,
    v: Tensor,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("GDN-2 backend must return (outputs, final_state)")
    outputs, final_state = result
    if not isinstance(outputs, Tensor) or not isinstance(final_state, Tensor):
        raise ValueError("GDN-2 backend outputs must both be tensors")
    expected_output = (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
    expected_state = (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    if outputs.shape != expected_output:
        raise ValueError(f"backend outputs must have shape {expected_output}, got {tuple(outputs.shape)}")
    if final_state.shape != expected_state:
        raise ValueError(
            f"backend final_state must have shape {expected_state}, got {tuple(final_state.shape)}"
        )
    if final_state.dtype != torch.float32:
        raise ValueError(f"backend final_state must be float32, got {final_state.dtype}")
    if outputs.device != q.device or final_state.device != q.device:
        raise ValueError("backend outputs must be on the input device")
    if initial_state is not None and initial_state.device != final_state.device:
        raise ValueError("initial_state and backend final_state must be on the same device")
    if not bool(torch.isfinite(outputs).all()) or not bool(torch.isfinite(final_state).all()):
        raise ValueError("backend outputs and final_state must be finite")
    return outputs, final_state


class OptimizedGDN2BackendAdapter:
    """Use a supplied optimized callable, otherwise use chunkwise PyTorch."""

    def __init__(
        self,
        optimized_callable: Callable[..., object] | None = None,
        fallback_backend: GDN2Backend | None = None,
    ) -> None:
        self.optimized_callable = optimized_callable
        self.fallback_backend = (
            fallback_backend if fallback_backend is not None else PyTorchChunkwiseGDN2Backend()
        )

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        log_decay: Tensor,
        erase_gate: Tensor,
        write_gate: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        callable_ = self.optimized_callable
        if callable(callable_):
            try:
                result = callable_(q, k, v, log_decay, erase_gate, write_gate, initial_state)
                return _validate_backend_result(result, q, v, initial_state)
            except (ImportError, NotImplementedError, TypeError):
                pass
        result = self.fallback_backend(
            q, k, v, log_decay, erase_gate, write_gate, initial_state
        )
        return _validate_backend_result(result, q, v, initial_state)


def _gradient_probe_loss(output: Tensor, state: Tensor) -> Tensor:
    return output.float().square().mean() + state.square().mean()


def assert_gdn2_backend_parity(
    backend: GDN2Backend,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_decay: Tensor,
    erase_gate: Tensor,
    write_gate: Tensor,
    initial_state: Tensor | None = None,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    check_gradients: bool = False,
    gradient_atol: float = 3e-5,
    gradient_rtol: float = 3e-5,
) -> None:
    """Raise when a candidate backend differs from the recurrent oracle.

    Value parity compares token outputs and the final FP32 state. Optional
    gradient parity reruns both paths on independent differentiable clones and
    compares gradients for q, k, v, log-decay, erase, write, and initial state.
    """

    reference_output, reference_state = gdn2_recurrent_reference(
        q, k, v, log_decay, erase_gate, write_gate, initial_state
    )
    candidate = backend(q, k, v, log_decay, erase_gate, write_gate, initial_state)
    output, state = _validate_backend_result(candidate, q, v, initial_state)
    torch.testing.assert_close(output, reference_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(state, reference_state, atol=atol, rtol=rtol)
    if not check_gradients:
        return

    source_tensors = (q, k, v, log_decay, erase_gate, write_gate)
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in source_tensors]
    candidate_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in source_tensors]
    reference_initial = (
        initial_state.detach().clone().requires_grad_(True) if initial_state is not None else None
    )
    candidate_initial = (
        initial_state.detach().clone().requires_grad_(True) if initial_state is not None else None
    )

    ref_output, ref_state = gdn2_recurrent_reference(*reference_inputs, reference_initial)
    candidate_output, candidate_state = backend(*candidate_inputs, candidate_initial)
    candidate_output, candidate_state = _validate_backend_result(
        (candidate_output, candidate_state), candidate_inputs[0], candidate_inputs[2], candidate_initial
    )
    ref_targets = reference_inputs + ([reference_initial] if reference_initial is not None else [])
    candidate_targets = candidate_inputs + ([candidate_initial] if candidate_initial is not None else [])
    reference_gradients = torch.autograd.grad(
        _gradient_probe_loss(ref_output, ref_state), ref_targets
    )
    candidate_gradients = torch.autograd.grad(
        _gradient_probe_loss(candidate_output, candidate_state), candidate_targets
    )
    for reference_gradient, candidate_gradient in zip(
        reference_gradients, candidate_gradients, strict=True
    ):
        torch.testing.assert_close(
            candidate_gradient,
            reference_gradient,
            atol=gradient_atol,
            rtol=gradient_rtol,
        )


class GatedDeltaNet2(nn.Module):
    """GDN-2 module with input/output ``[B,T,D]`` and recurrent cache."""

    def __init__(self, config: ModelConfig, backend: GDN2Backend | None = None) -> None:
        super().__init__()
        self.config = config
        if config.gdn_num_key_heads != config.gdn_num_value_heads:
            raise ValueError("GDN key and value head counts must match")
        if config.gdn_key_dim != config.gdn_value_dim:
            raise ValueError("GDN key and value dimensions must match")
        if config.gdn_num_key_heads * config.gdn_key_dim != config.d_model:
            raise ValueError("GDN key/value width must equal d_model")
        if config.gdn_conv_kernel_size <= 0:
            raise ValueError("gdn_kernel_size must be positive")

        self.d_model = config.d_model
        self.n_heads = config.gdn_num_key_heads
        self.key_dim = config.gdn_key_dim
        self.value_dim = config.gdn_value_dim
        self.kernel_size = config.gdn_conv_kernel_size
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_conv = nn.Conv1d(
            self.d_model, self.d_model, self.kernel_size, groups=self.d_model, bias=False
        )
        self.k_conv = nn.Conv1d(
            self.d_model, self.d_model, self.kernel_size, groups=self.d_model, bias=False
        )
        self.v_conv = nn.Conv1d(
            self.d_model, self.d_model, self.kernel_size, groups=self.d_model, bias=False
        )
        self.erase_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.decay_proj = nn.Sequential(
            nn.Linear(self.d_model, 64, bias=False),
            nn.Linear(64, self.d_model, bias=False),
        )
        self.output_gate = nn.Sequential(
            nn.Linear(self.d_model, 64, bias=False),
            nn.Linear(64, self.d_model, bias=True),
        )
        nn.init.zeros_(self.output_gate[-1].bias)
        self.output_norm = RMSNorm(self.value_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads, self.key_dim, dtype=torch.float32))
        with torch.no_grad():
            self.A_log.copy_(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))
            dt = torch.logspace(-3, -1, self.key_dim, dtype=torch.float32)
            self.dt_bias.copy_(torch.log(torch.expm1(dt)).expand(self.n_heads, -1))
        self.backend: GDN2Backend = (
            backend
            if backend is not None
            else PyTorchChunkwiseGDN2Backend(config.gdn_chunk_size)
        )

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> nn.Module:
        module = super()._apply(fn)
        self.A_log.data = self.A_log.data.float()
        self.dt_bias.data = self.dt_bias.data.float()
        return module

    def _causal_depthwise(
        self, raw: Tensor, history: Tensor, convolution: nn.Conv1d
    ) -> Tensor:
        combined = torch.cat((history, raw), dim=1)
        filtered = convolution(combined.transpose(1, 2)).transpose(1, 2)
        return F.silu(filtered)

    def _validate_cache(
        self, cache: GDN2Cache, batch: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        state_shape = (batch, self.n_heads, self.key_dim, self.value_dim)
        if cache.recurrent_state.shape != state_shape:
            raise ValueError(
                f"cache recurrent_state must have shape {state_shape}, "
                f"got {tuple(cache.recurrent_state.shape)}"
            )
        history_shape = (batch, self.kernel_size - 1, self.d_model)
        for name, history in (
            ("q_history", cache.q_history),
            ("k_history", cache.k_history),
            ("v_history", cache.v_history),
        ):
            if history.shape != history_shape:
                raise ValueError(f"cache {name} must have shape {history_shape}, got {tuple(history.shape)}")
            if history.device != device:
                raise ValueError(f"cache {name} must be on the input device")
            if history.dtype != dtype:
                raise ValueError(f"cache {name} must have the input dtype")
        if cache.recurrent_state.device != device:
            raise ValueError("cache recurrent_state must be on the input device")
        if cache.recurrent_state.dtype != torch.float32:
            raise ValueError("cache recurrent_state must be float32")
        if not bool(torch.isfinite(cache.recurrent_state).all()):
            raise ValueError("cache recurrent_state must be finite")
        if any(
            not bool(torch.isfinite(history).all())
            for history in (cache.q_history, cache.k_history, cache.v_history)
        ):
            raise ValueError("cache histories must be finite")

    def forward(
        self, x: Tensor, cache: GDN2Cache | None = None, return_cache: bool = False
    ) -> Tensor | tuple[Tensor, GDN2Cache]:
        """Process ``x`` of shape ``[B,T,D]`` and optionally return its cache."""

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected input [batch, sequence, {self.d_model}], got {tuple(x.shape)}")
        if x.shape[1] == 0:
            raise ValueError("GDN-2 requires a non-empty sequence")
        if x.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length cannot exceed max_seq_len")
        batch = x.shape[0]
        if cache is None:
            state = torch.zeros(
                batch,
                self.n_heads,
                self.key_dim,
                self.value_dim,
                device=x.device,
                dtype=torch.float32,
            )
            zero_history = x.new_zeros((batch, self.kernel_size - 1, self.d_model))
            q_history = k_history = v_history = zero_history
        else:
            self._validate_cache(cache, batch, x.device, x.dtype)
            state = cache.recurrent_state
            q_history, k_history, v_history = cache.q_history, cache.k_history, cache.v_history

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        q = self._causal_depthwise(q_raw, q_history, self.q_conv)
        k = self._causal_depthwise(k_raw, k_history, self.k_conv)
        v = self._causal_depthwise(v_raw, v_history, self.v_conv)
        q = F.normalize(q.view(batch, x.shape[1], self.n_heads, self.key_dim), dim=-1, eps=1e-6)
        k = F.normalize(k.view(batch, x.shape[1], self.n_heads, self.key_dim), dim=-1, eps=1e-6)
        v = v.view(batch, x.shape[1], self.n_heads, self.value_dim)

        decay_features = self.decay_proj(x).float().view(
            batch, x.shape[1], self.n_heads, self.key_dim
        )
        log_decay = -torch.exp(self.A_log.float()).view(1, 1, self.n_heads, 1) * F.softplus(
            decay_features + self.dt_bias.float().view(1, 1, self.n_heads, self.key_dim)
        )
        erase_gate = torch.sigmoid(self.erase_proj(x)).view(
            batch, x.shape[1], self.n_heads, self.key_dim
        )
        write_gate = torch.sigmoid(self.write_proj(x)).view(
            batch, x.shape[1], self.n_heads, self.value_dim
        )
        recurrent, final_state = self.backend(
            q,
            k,
            v,
            log_decay,
            erase_gate,
            write_gate,
            state,
        )
        recurrent, final_state = _validate_backend_result(
            (recurrent, final_state), q, v, state
        )
        mixed = self.output_norm(recurrent)
        output_gate = F.silu(self.output_gate(x)).view(
            batch, x.shape[1], self.n_heads, self.value_dim
        )
        output = self.out_proj((mixed * output_gate).reshape(batch, x.shape[1], self.d_model))
        if not return_cache:
            return output
        history_length = self.kernel_size - 1
        histories = [
            torch.cat((history, raw), dim=1)[:, -history_length:, :]
            if history_length
            else raw[:, :0]
            for history, raw in ((q_history, q_raw), (k_history, k_raw), (v_history, v_raw))
        ]
        return output, GDN2Cache(final_state, histories[0], histories[1], histories[2])


__all__ = [
    "GDN2Backend",
    "GDN2Cache",
    "GatedDeltaNet2",
    "OptimizedGDN2BackendAdapter",
    "PyTorchChunkwiseGDN2Backend",
    "PyTorchGDN2Backend",
    "assert_gdn2_backend_parity",
    "gdn2_chunkwise_reference",
    "gdn2_recurrent_reference",
]
