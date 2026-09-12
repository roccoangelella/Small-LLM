"""Opt-in ``torch.compile`` execution lane for the MoE decoder blocks.

The lane is *execution*, never identity. It changes no parameter, no buffer, no
state-dict key and no checkpoint field: ``nn.Module.compile()`` installs a
compiled ``_compiled_call_impl`` on the module it is called on and leaves the
module object, its children and its ``state_dict()`` untouched. That is why the
mode lives in the trainer-side CLI and not in :class:`MOE_model.config.MoEModelConfig`.

Scope
-----
``blocks`` compiles every :class:`~MOE_model.model.MoEDecoderBlock` in place, so
one compiled region spans the norms, the mixer (GDN-2 or gated MHA), the router
scoring, the padded expert SwiGLU GEMM and the recombination. Blocks are the
largest unit that still lets the mixer's unavoidable breaks stay local.

The lane targets the accepted geometry's ``dropless_padded_batched_gemm``
dispatch. It still *runs* correctly on the version-2 grouped dispatch
(``DroplessTop1MoE``, and ``DroplessTopKMoE`` with ``batched=False``), but that
path slices one Python loop iteration per expert, so Dynamo re-specialises on
every distinct expert-group size and reaches ``recompile_limit`` before falling
back to eager for that frame — measured on a smoke launch. It is not refused,
because it is a warm-up cost rather than an error, but ``--compile blocks`` buys
nothing there.

Dynamic capacity
----------------
``DroplessTopKMoE`` reads ``capacity = int(counts.max())`` once per microbatch.
Dynamo cannot trace that host synchronisation with the default
``capture_scalar_outputs=False``, so it breaks the graph there and the padded
buffer, the three ``bmm`` calls and the recombination land in the *following*
graph with ``capacity`` as an ordinary Python integer. Compiling with
``dynamic=True`` makes that integer a dynamic symbol from the first compile, so a
per-microbatch varying capacity is measured not to recompile (see
``tests/test_moe_compile_lane.py``). Leaving ``capture_scalar_outputs`` at its
default is deliberate: turning it on makes ``capacity`` an *unbacked* symbol and
Inductor then fails to lower ``torch.bmm`` on the padded buffer with
``GuardOnDataDependentSymNode: Could not guard on ... Eq(u0, 1)``. No
``torch._dynamo.config`` setting is required or applied by this lane.

Quantile Balancing
------------------
``SwitchTopKRouter.begin_step`` / ``_observe_scores`` / ``commit_load`` are
``no_grad`` bookkeeping over host-side Python state (the score frontier and the
target rank) that decides the selection bias. Tracing them would let Dynamo
specialise on that state, so they are excluded with ``torch.compiler.disable``
and keep running in eager exactly as before — the selection bias is therefore
bit-identical with the lane on. ``_observe_scores`` is called from inside the
compiled block, so its exclusion is what splits the router graph; the other two
are called from :mod:`MOE_model.step`, outside any compiled region, and are
excluded defensively.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .model import MoEDecoderBlock
from .router import SwitchTopKRouter

COMPILE_MODES: tuple[str, ...] = ("off", "blocks")
DEFAULT_COMPILE_MODE = "off"

#: Router methods kept in eager so Quantile Balancing semantics cannot change.
ROUTER_BOOKKEEPING_METHODS: tuple[str, ...] = (
    "begin_step",
    "_observe_scores",
    "commit_load",
)

_DISABLE_MARKER = "_moe_compile_lane_disabled"


def exclude_router_bookkeeping() -> None:
    """Mark the Quantile Balancing bookkeeping as never-compiled. Idempotent.

    The methods are wrapped on :class:`~MOE_model.router.SwitchTopKRouter` itself,
    which is what ``SwitchTop1Router`` inherits, so every router in the process is
    covered by one wrap. ``torch.compiler.disable`` is semantically transparent in
    eager execution: it only tells Dynamo to stop tracing.
    """

    for name in ROUTER_BOOKKEEPING_METHODS:
        function = getattr(SwitchTopKRouter, name)
        if getattr(function, _DISABLE_MARKER, False):
            continue
        disabled = torch.compiler.disable(function)
        setattr(disabled, _DISABLE_MARKER, True)
        setattr(SwitchTopKRouter, name, disabled)


def router_bookkeeping_is_excluded() -> bool:
    return all(
        getattr(getattr(SwitchTopKRouter, name), _DISABLE_MARKER, False)
        for name in ROUTER_BOOKKEEPING_METHODS
    )


def decoder_blocks(model: nn.Module) -> Iterable[MoEDecoderBlock]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("the compile lane needs a model exposing `blocks`")
    return blocks


def block_is_compiled(block: nn.Module) -> bool:
    return getattr(block, "_compiled_call_impl", None) is not None


def compiled_block_count(model: nn.Module) -> int:
    return sum(1 for block in decoder_blocks(model) if block_is_compiled(block))


def apply_compile_lane(
    model: nn.Module,
    mode: str,
    *,
    backend: str | None = None,
    dynamic: bool | None = True,
) -> None:
    """Apply the requested compile mode to ``model`` in place.

    ``off`` is a strict no-op. ``blocks`` compiles each decoder block in place
    with ``nn.Module.compile``, which never rebinds parameters and never changes
    ``state_dict()`` keys, so an optimizer built afterwards captures exactly the
    same parameter objects and checkpoints stay compatible in both directions.

    ``backend=None`` keeps ``torch.compile``'s own default (Inductor); tests pass
    ``"aot_eager"`` to keep CPU compile time bounded.
    """

    if not isinstance(mode, str) or mode not in COMPILE_MODES:
        raise ValueError(
            f"unsupported compile mode {mode!r}; expected one of {COMPILE_MODES}"
        )
    if mode == "off":
        return

    exclude_router_bookkeeping()
    options: dict[str, object] = {"dynamic": dynamic}
    if backend is not None:
        options["backend"] = backend
    for block in decoder_blocks(model):
        if not isinstance(block, MoEDecoderBlock):
            raise TypeError(
                "the compile lane compiles MoEDecoderBlock instances; "
                f"found {type(block).__name__}"
            )
        if block_is_compiled(block):
            continue
        block.compile(**options)


__all__ = [
    "COMPILE_MODES",
    "DEFAULT_COMPILE_MODE",
    "ROUTER_BOOKKEEPING_METHODS",
    "apply_compile_lane",
    "block_is_compiled",
    "compiled_block_count",
    "decoder_blocks",
    "exclude_router_bookkeeping",
    "router_bookkeeping_is_excluded",
]
