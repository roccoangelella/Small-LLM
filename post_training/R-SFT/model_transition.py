"""Checkpoint-compatible transition from S0 to the R-SFT semantic vocabulary.

R-SFT promotes three rows that already exist inside the padded tied embedding.
The physical embedding shape never changes: pretrained/S0 rows are copied
exactly, IDs 50_257..50_259 are reinitialized with the frozen GPT-style
Normal(0, 0.02) policy, and the remaining padded rows stay zero.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib.util
import sys
from types import ModuleType

import torch

from model.config import ModelConfig
from model.model import SmallLLM


def _load_tokenizer_contract() -> ModuleType:
    module_name = "small_llm_rsft_tokenizer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("tokenizer.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load R-SFT tokenizer module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


tokenizer = _load_tokenizer_contract()

BASE_SEMANTIC_VOCAB_SIZE = tokenizer.BASE_SEMANTIC_VOCAB_SIZE
R_SFT_SEMANTIC_VOCAB_SIZE = tokenizer.R_SFT_SEMANTIC_VOCAB_SIZE
REASONING_START_TOKEN_ID = tokenizer.REASONING_START_TOKEN_ID
ANSWER_START_TOKEN_ID = tokenizer.ANSWER_START_TOKEN_ID
PROMOTED_TOKEN_COUNT = R_SFT_SEMANTIC_VOCAB_SIZE - BASE_SEMANTIC_VOCAB_SIZE
PROMOTED_ROW_STD = 0.02


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [device.index]
    return [torch.cuda.current_device()]


def promote_s0_model_for_rsft(
    parent_model: SmallLLM,
    parent_config: ModelConfig,
    *,
    seed: int = 17,
) -> tuple[SmallLLM, ModelConfig]:
    """Return an R-SFT model with only the three promoted rows newly initialized.

    The parent is left untouched.  All non-promoted parameters and the original
    50,257 semantic embedding rows are copied bit-for-bit.  The function also
    preserves the caller's RNG state, so promotion does not perturb later data
    or training randomness.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if parent_config.semantic_vocab_size != BASE_SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            f"R-SFT promotion requires an S0 semantic vocabulary of {BASE_SEMANTIC_VOCAB_SIZE}"
        )
    if parent_config.padded_vocab_size < R_SFT_SEMANTIC_VOCAB_SIZE:
        raise ValueError("the parent padded vocabulary has no rows available for R-SFT promotion")
    if parent_model.config != parent_config:
        raise ValueError("parent model/config mismatch")

    parent_parameter = next(parent_model.parameters(), None)
    if parent_parameter is None:
        raise ValueError("parent model has no parameters")
    device = parent_parameter.device
    dtype = parent_parameter.dtype
    rsft_config = replace(parent_config, semantic_vocab_size=R_SFT_SEMANTIC_VOCAB_SIZE)

    # Construction consumes randomness even though every parameter is about to
    # be overwritten from the parent. Keep that consumption invisible to the
    # caller, then reset to the run seed immediately before initializing the
    # only genuinely new semantic rows.
    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(seed)
        rsft_model = SmallLLM(rsft_config).to(device=device, dtype=dtype)
        rsft_model.load_state_dict(parent_model.state_dict(), strict=True)

        weight = rsft_model.token_embedding.weight
        if weight.shape[0] != parent_config.padded_vocab_size:
            raise RuntimeError("R-SFT promotion changed the physical embedding shape")

        torch.manual_seed(seed)
        with torch.no_grad():
            weight[
                BASE_SEMANTIC_VOCAB_SIZE:R_SFT_SEMANTIC_VOCAB_SIZE
            ].normal_(mean=0.0, std=PROMOTED_ROW_STD)
            weight[R_SFT_SEMANTIC_VOCAB_SIZE:].zero_()

    if not torch.equal(
        rsft_model.token_embedding.weight[:BASE_SEMANTIC_VOCAB_SIZE],
        parent_model.token_embedding.weight[:BASE_SEMANTIC_VOCAB_SIZE],
    ):
        raise RuntimeError("R-SFT promotion modified pretrained semantic embedding rows")
    if torch.count_nonzero(
        rsft_model.token_embedding.weight[R_SFT_SEMANTIC_VOCAB_SIZE:]
    ).item() != 0:
        raise RuntimeError("R-SFT promotion left non-semantic padded rows nonzero")

    return rsft_model, rsft_config


__all__ = [
    "BASE_SEMANTIC_VOCAB_SIZE",
    "PROMOTED_ROW_STD",
    "PROMOTED_TOKEN_COUNT",
    "R_SFT_SEMANTIC_VOCAB_SIZE",
    "promote_s0_model_for_rsft",
]
