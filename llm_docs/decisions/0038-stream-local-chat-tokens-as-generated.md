---
status: accepted
date: 2026-08-11
supersedes: null
---

# 0038 — Stream local chat tokens as generated

## Context and problem statement

The root-level `chat.py` initially waited for the complete sampled response before printing it. Local inference can have noticeable time-to-first-token because the first prompt requires a full model forward pass, a cold CUDA process may also pay FLA/Triton JIT startup cost, and the current hybrid GDN-2/MHA model has no unified cached decode contract. Waiting for the entire answer made that compute latency feel substantially worse than necessary.

## Considered options

- Keep buffered whole-response printing.
- Stream each sampled token immediately while preserving the existing sampling semantics.
- First implement a unified GDN-2/MHA inference cache, then add streaming.

## Decision outcome

Chosen option: **stream each sampled token immediately while preserving the current sampler and model execution path**. `chat.py` flushes `assistant> ` before generation and prints decoded text after every sampled non-EOS token. GPT-2 token bytes are passed through an incremental UTF-8 decoder so code points split across token boundaries remain correct.

Streaming is explicitly a presentation improvement, not an inference-cache implementation. Time-to-first-token and per-token compute remain governed by the current full-prefix forward path.

## Consequences

### Positive

- Users see output as soon as the first token is sampled instead of waiting for the full response.
- Temperature, top-k, top-p, seed, EOS handling, prompt serialization, and checkpoint selection remain unchanged.
- Split UTF-8 sequences are rendered correctly rather than decoding every token independently with replacement characters.

### Negative or limiting

- Streaming does not reduce the first prompt forward latency.
- The current model still recomputes the retained prefix for every generated token because unified cached decoding is not implemented.
- A cold CUDA environment may still pause before the first streamed token while required FLA/Triton kernels are compiled or loaded from cache.

## Validation

- Unit-test incremental decoding with a multi-byte UTF-8 character split across multiple synthetic token byte sequences.
- Run the local chat CLI against a completed SFT checkpoint and verify output appears token-by-token with no duplicated buffered response at turn completion.
- Treat a future unified cached-decode implementation as a separate performance decision and qualification task.

## Links

- [`../../chat.py`](../../chat.py)
- [`../runbooks/local_sft_chat.md`](../runbooks/local_sft_chat.md)
- [`0036-add-local-completed-sft-chat-cli.md`](0036-add-local-completed-sft-chat-cli.md)
