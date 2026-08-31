# 0132 — Canonicalize the 100M/2B 10% SFT peak-through-3000 run

Date: 2026-08-31
Status: Accepted

## Decision

Treat `100m-2b-sft-s0-10pct-peak3000-001` as the standard 100M/2B SFT model whenever the project refers to evaluating, qualifying, or chatting with the 100M/2B 10% SFT trajectory.

The historical `100m-2b-sft-s0-10pct-longpeak-001` run remains an experiment artifact, but it is no longer the default selected by the canonical Kaggle SFT launcher for `--model 100M --tokens 2B --sft-fraction 10%`.

## Rationale

The previous resolver path still mapped the 100M/2B 10% SFT fraction to the abandoned `10pct-longpeak` identity. Training later redirected the capacity-aware trajectory to `100m-2b-sft-s0-10pct-peak3000-001`, but evaluation inherited the stale resolver identity and could evaluate `latest` from the wrong run.

The peak-through-3000 run is the completed trajectory we want to treat as the standard model for follow-up evaluation and manual chat checks.

## Implementation

- `kaggle/sft_cli.py` now resolves the 100M/2B 10% SFT profile to `100m-2b-sft-s0-10pct-peak3000-001` with the `peak-through-3000` W&B name suffix.
- `chat.py` now defaults `python chat.py --model_params 100M --num_tokens 2B --sft` to `100m-2b-sft-s0-10pct-peak3000-001`.
- Tests were updated so both the SFT launcher resolver and chat profile registry fail if they regress to the historical `10pct-longpeak` or 4% SFT defaults.

## Operational consequence

The existing eval command:

```bash
python kaggle/launch_sft.py eval \
  --model 100M \
  --tokens 2B \
  --sft-fraction 10% \
  --parent-repo-id roccoangelella/small-llm-100m-qualification \
  --checkpoint-repo-id roccoangelella/small-llm-100m-qualification \
  --suite full \
  --device cuda \
  --precision fp16 \
  --batch-size 1 \
  --validation-blocks 32 \
  --test-blocks 32 \
  --output /kaggle/working/100m-2b-sft-s0-10pct-full-qualification.json
```

now resolves its SFT checkpoint lookup to:

```text
--sft-run-id 100m-2b-sft-s0-10pct-peak3000-001
--sft-pointer latest
```

rather than the stale `100m-2b-sft-s0-10pct-longpeak-001` run.
