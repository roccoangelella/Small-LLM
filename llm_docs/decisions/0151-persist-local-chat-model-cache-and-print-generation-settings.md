# 0151 — Persist local chat model cache and print generation settings

## Decision

`chat.py` must keep downloaded chat artifacts on disk instead of storing them in a temporary directory that is deleted when the process exits.

The persistent cache lives under the repository root at:

```text
chat_models/<stage>/<run_id>/
```

Each cached model keeps its verified checkpoint tree plus small cache metadata identifying the repository, run, stage, source, and checkpoint. A subsequent launch for the same profile must reuse the cached checkpoint after running the normal local manifest/completion/tokenizer-stage verification. If a cached copy fails verification, `chat.py` may discard that managed cache entry and perform one clean download again.

`chat_models/` is local runtime state and must be ignored by Git.

At startup, after loading the model, `chat.py` must print the effective generation configuration used by the interactive sampler, including at least temperature, top-p, top-k, maximum new tokens, seed policy, model context length, EOS token ID, and precision.

This decision does not change the current sampling values. In particular, local chat remains `temperature=1.0`, `top_p=1.0`, `top_k=50`, `MAX_NEW_TOKENS=128`, and base seed `17` until a separate decision changes them.

## Rationale

The previous implementation used `tempfile.TemporaryDirectory`, then explicitly cleaned it when the chat exited. That made every new local chat process download the same completed checkpoint again even when the user had just used it.

Persistent verified caching removes unnecessary network transfer and startup time while preserving the existing integrity/completion gates. Printing the effective sampler configuration makes qualitative chat sessions reproducible and immediately auditable.
