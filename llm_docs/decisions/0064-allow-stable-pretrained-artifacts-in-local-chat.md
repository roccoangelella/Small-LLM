---
status: accepted
date: 2026-08-13
supersedes: 0036
---

# 0064 — Allow stable pretrained artifacts in local chat

## Context and problem statement

The local `chat.py` originally accepted only explicitly registered completed SFT trajectories. The completed 100M / 2B endpoint is different: it is a stable pretrained model artifact published under the canonical `models/100m-2b-data-001/...` namespace, and there is no registered `100m-2b-sft-s0-001` trajectory.

The user wants the completed 100M / 2B run selectable from the same local interactive CLI. Inventing an SFT run ID would violate the fail-closed artifact contract and would misrepresent the model as instruction-tuned.

## Considered options

- Register a nonexistent `100m-2b-sft-s0-001` run and let download fail later.
- Keep `chat.py` SFT-only and refuse the completed 100M / 2B pretrained artifact.
- Expand `chat.py` to support explicitly registered completed artifacts of two types: completed SFT trajectories and stable pretrained model artifacts.

## Decision outcome

Chosen option: **expand the local chat CLI to explicitly registered completed artifact types, adding 100M / 2B as the stable pretrained run `100m-2b-data-001`.**

The command is:

```bash
python chat.py --model_params 100M --num_tokens 2B
```

Existing 20M SFT profiles retain their current path and strict SFT identity/completion checks. The 100M / 2B profile uses `trainer.model_artifact.download_verified_model_artifact`, reads only `SMALL_LLM_HF_REPO_ID`, verifies the stable artifact manifest, and requires the embedded WSD trainer state to be complete before loading weights.

The interactive serialization remains the existing chat template for consistency of the CLI surface. This does **not** make the 100M / 2B model instruction-tuned; its responses are those of the pretrained base model under that prompt formatting.

## Consequences

### Positive

- The completed 100M / 2B model can be inspected interactively with the same local CLI.
- The implementation reuses the canonical stable-model artifact transport instead of inventing a new checkpoint path.
- Existing SFT runs remain fail-closed on SFT stage identity and schedule completion.
- A separately configured SFT repository cannot accidentally redirect the 100M / 2B base-model lookup; stable pretrained artifacts use `SMALL_LLM_HF_REPO_ID`.

### Negative or limiting

- The 100M / 2B run is a pretrained base model, so chat-style behavior should not be interpreted as SFT capability.
- The CLI still has no unified cached GDN-2/MHA decode path and remains a qualitative convenience surface rather than a serving benchmark.

## Validation

- `tests/test_chat_cli.py` must resolve 100M / 2B to `100m-2b-data-001` with the stable-model source and keep unknown profiles fail-closed.
- A live invocation should resolve the canonical stable artifact under `models/100m-2b-data-001/...`, verify it, and load the completed 100M / 2B weights.

## Links

- [`0044-publish-100m-2b-final-model-to-hugging-face.md`](0044-publish-100m-2b-final-model-to-hugging-face.md)
- [`../runbooks/local_sft_chat.md`](../runbooks/local_sft_chat.md)
- [`../current/status.md`](../current/status.md)
