"""Readable PyTorch GDN-2 recurrence and backend boundary."""

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
    """Run recurrence on ``[B,T,H,K]`` q/k and ``[B,T,H,V]`` v tensors."""

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
    """Reference PyTorch backend for the GDN-2 recurrence."""

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
    """Use a supplied recurrence callable, otherwise use the PyTorch reference."""

    def __init__(self, optimized_callable: Callable[..., object] | None = None) -> None:
        self.optimized_callable = optimized_callable
        self.reference_backend = PyTorchGDN2Backend()

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
        result = self.reference_backend(q, k, v, log_decay, erase_gate, write_gate, initial_state)
        return _validate_backend_result(result, q, v, initial_state)


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
) -> None:
    """Raise if a candidate recurrence backend differs from the oracle.

    Backend authors call this qualification helper before installing an
    optimized path.  It compares both token outputs and the final FP32 state,
    so a shape-valid but mathematically different kernel cannot pass merely by
    satisfying the adapter boundary.
    """

    reference_output, reference_state = gdn2_recurrent_reference(
        q, k, v, log_decay, erase_gate, write_gate, initial_state
    )
    candidate = backend(q, k, v, log_decay, erase_gate, write_gate, initial_state)
    output, state = _validate_backend_result(candidate, q, v, initial_state)
    torch.testing.assert_close(output, reference_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(state, reference_state, atol=atol, rtol=rtol)


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
            # Keep these state-dynamics parameters in their reference-style
            # positive-rate / inverse-softplus parameterization.  Ordinary
            # initializer experiments deliberately preserve them.
            self.A_log.copy_(torch.log(torch.arange(1, self.n_heads + 1, dtype=torch.float32)))
            dt = torch.logspace(-3, -1, self.key_dim, dtype=torch.float32)
            self.dt_bias.copy_(torch.log(torch.expm1(dt)).expand(self.n_heads, -1))
        self.backend: GDN2Backend = backend if backend is not None else PyTorchGDN2Backend()

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
        if any(not bool(torch.isfinite(history).all()) for history in (cache.q_history, cache.k_history, cache.v_history)):
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

        decay_features = self.decay_proj(x).float().view(batch, x.shape[1], self.n_heads, self.key_dim)
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
