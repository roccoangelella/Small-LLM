---
status: accepted
date: 2026-08-08
supersedes: null
---

# 0019 — Resume the active 500M trajectory with FLA GDN-2 execution

## Context

The integrated Small-LLM FLA GDN-2 layer passed the user-run Kaggle full-layer forward/backward parity probe. The active 20M/500M run already has verified remote checkpoints created under the historical `gdn_chunk_size=32` configuration and the old adaptive PyTorch execution backend.

The 500M one-click launcher is fail-closed and creates a detached training worktree at its pinned implementation commit. Therefore changing `model/` on `main` alone would not migrate the actual training subprocess; the launcher pin must move to an implementation commit containing the qualified FLA adapter.

Changing the trainer CLI from `--gdn-chunk-size 32` to 64 is not acceptable for resume because trainer checkpoints serialize the model configuration and restore checks it strictly. Such a change would reject the existing checkpoint even though chunk size is only an execution grouping.

## Decision

The normal 20M/500M one-click command is authorized to restore the latest verified 500M checkpoint and continue it with the qualified FLA GDN-2 CUDA execution backend.

Preserve the historical checkpoint/model configuration unchanged:

```text
gdn_chunk_size = 32
```

On CUDA, the integrated backend may nevertheless evaluate the same recurrence with FLA's fixed 64-token GDN-2 kernel. The configured value 32 remains the adaptive/reference fallback chunk size and part of strict checkpoint identity; it is not rewritten during restore.

The 500M entry point is repinned to implementation commit:

```text
a1471472ca9b5d07f70c844460acffe5c96c5200
```

That worktree contains the checkpoint-compatible FLA adapter, the model runtime dependency on `fla-core==0.5.1`, the chunk-32/FLA64 compatibility probe, and unchanged model parameter/state-dict structure.

## Resume behavior

Running the ordinary command:

```bash
python kaggle/run_20m_500m.py
```

continues to:

1. restore only the latest verified remote checkpoint under the existing 500M run identity;
2. verify its dataset/Drive-manifest cursor;
3. strict-load the same model configuration, model weights, optimizer, scheduler, scaler, RNG state, consumed-token position, and WSD position;
4. continue from the next optimizer update;
5. use FLA GDN-2 for CUDA recurrence execution after restore.

No checkpoint tensor conversion is required because the backend adds no learned parameters or state-dict entries.

## Scientific consequence

This is an explicit implementation migration within the existing 500M trajectory. The mathematical GDN-2 recurrence is intended to remain the same, but exact bitwise replay equivalence with the hypothetical all-adaptive continuation is no longer expected because floating-point operation ordering changes.

The resumed trajectory remains useful for observing checkpoint behavior and practical continued learning. A later fresh 500M run from update 1 with FLA remains a separate decision if a clean single-backend scientific reference is desired.

## Links

- [`../../kaggle/run_20m_500m.py`](../../kaggle/run_20m_500m.py)
- [`../../model/gdn2_fla.py`](../../model/gdn2_fla.py)
- [`../evidence/gdn2_fla_layer_integration_2026-08-08.md`](../evidence/gdn2_fla_layer_integration_2026-08-08.md)
- [`0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md`](0018-integrate-fla-gdn2-as-checkpoint-compatible-cuda-backend.md)
