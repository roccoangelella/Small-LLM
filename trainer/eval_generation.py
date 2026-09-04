"""Batched autoregressive generation for evaluation-only workloads.

The project generation contract is unchanged: each request keeps its own prompt,
maximum continuation budget, sampling parameters, and RNG seed.  This module
only batches independent model forwards and restores results to request order.
Right padding is safe because logits are read at each sequence's last real token
under causal decoding; padding therefore occurs strictly after the scored prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from trainer.post_pretraining_prompt_suite import _autocast_context, _filter_logits

DEFAULT_GENERATION_BATCH_SIZE = 16


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    seed: int


def _validate_request(request: GenerationRequest, *, max_seq_len: int) -> None:
    if not request.prompt_ids:
        raise ValueError("generation prompt must contain at least one token")
    if request.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")


def _report_progress(label: str | None, completed: int, total: int, previous: int) -> int:
    if label is None or total <= 0:
        return previous
    percent = int(completed * 100 / total)
    bucket = min(10, percent // 10)
    if completed == total or bucket > previous:
        print(f"[{label}] {completed}/{total} requests complete ({percent}%)", flush=True)
        return bucket
    return previous


@torch.inference_mode()
def _generate_chunk(
    model: nn.Module,
    requests: Sequence[GenerationRequest],
    *,
    max_seq_len: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    top_k: int,
    precision: str,
) -> list[list[int]]:
    device = next(model.parameters()).device
    sequences = [list(request.prompt_ids) for request in requests]
    generated: list[list[int]] = [[] for _ in requests]
    finished = [request.max_new_tokens == 0 for request in requests]
    generators: list[torch.Generator] = []
    for request in requests:
        generator = torch.Generator(device=device)
        generator.manual_seed(request.seed)
        generators.append(generator)

    while not all(finished):
        active = [index for index, done in enumerate(finished) if not done]
        windows = [sequences[index][-max_seq_len:] for index in active]
        width = max(len(window) for window in windows)
        input_ids = torch.zeros((len(active), width), dtype=torch.long, device=device)
        last_positions = torch.empty(len(active), dtype=torch.long, device=device)
        for row, window in enumerate(windows):
            input_ids[row, : len(window)] = torch.tensor(window, dtype=torch.long, device=device)
            last_positions[row] = len(window) - 1

        with _autocast_context(device, precision):
            logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise RuntimeError("evaluation generation model must return [batch, time, vocab] logits")
        row_ids = torch.arange(len(active), device=device)
        next_logits = logits[row_ids, last_positions, :].float()

        if temperature == 0:
            next_tokens = next_logits.argmax(dim=-1)
        else:
            filtered = _filter_logits(next_logits / temperature, top_k=top_k, top_p=top_p)
            probabilities = torch.softmax(filtered, dim=-1)
            if not torch.isfinite(probabilities).all() or bool((probabilities.sum(dim=-1) <= 0).any()):
                raise FloatingPointError("sampling produced an invalid probability distribution")
            sampled = []
            for row, request_index in enumerate(active):
                token = torch.multinomial(
                    probabilities[row : row + 1],
                    1,
                    generator=generators[request_index],
                )
                sampled.append(token.reshape(()))
            next_tokens = torch.stack(sampled)

        for row, request_index in enumerate(active):
            token_id = int(next_tokens[row].item())
            generated[request_index].append(token_id)
            sequences[request_index].append(token_id)
            request = requests[request_index]
            if token_id == eos_token_id or len(generated[request_index]) >= request.max_new_tokens:
                finished[request_index] = True

    return generated


@torch.inference_mode()
def sample_token_ids_batched(
    model: nn.Module,
    requests: Sequence[GenerationRequest],
    *,
    max_seq_len: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
    top_k: int,
    precision: str,
    batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    progress_label: str | None = None,
) -> list[list[int]]:
    """Generate independent requests in length-bucketed mini-batches.

    Per-request generators preserve the legacy seed contract.  Results are
    restored to the caller's original order even though requests are sorted by
    prompt length for lower padding overhead.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    for request in requests:
        _validate_request(request, max_seq_len=max_seq_len)
    if not requests:
        return []

    indexed = sorted(
        enumerate(requests),
        key=lambda item: (len(item[1].prompt_ids), item[1].max_new_tokens, item[0]),
    )
    results: list[list[int] | None] = [None] * len(requests)
    completed = 0
    progress_bucket = -1
    progress_bucket = _report_progress(progress_label, completed, len(requests), progress_bucket)

    for start in range(0, len(indexed), batch_size):
        batch = indexed[start : start + batch_size]
        batch_requests = [request for _, request in batch]
        batch_results = _generate_chunk(
            model,
            batch_requests,
            max_seq_len=max_seq_len,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            precision=precision,
        )
        for (original_index, _), generated in zip(batch, batch_results, strict=True):
            results[original_index] = generated
        completed += len(batch)
        progress_bucket = _report_progress(progress_label, completed, len(requests), progress_bucket)

    if any(result is None for result in results):
        raise RuntimeError("batched generation failed to populate every request")
    return [list(result) for result in results if result is not None]


__all__ = [
    "DEFAULT_GENERATION_BATCH_SIZE",
    "GenerationRequest",
    "sample_token_ids_batched",
]
