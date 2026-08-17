# Decision 0084: Select the 100M/2B SFT chat artifact explicitly

- Status: Accepted
- Date: 2026-08-15

## Context and problem statement

`chat.py --model_params 100M --num_tokens 2B` was intentionally mapped to the stable pretrained `100m-2b-data-001` artifact, while the completed 100M/2B SFT run is `100m-2b-sft-s0-001`. Repointing the existing command implicitly would make it harder to access the pretrained baseline and would make model-stage selection ambiguous.

The 100M/2B SFT run can also use the streamed `torch.save` trainer-state format introduced for low-memory DDP checkpoint durability, whereas `chat.py` historically loaded `trainer_state.pkl` directly with `pickle.load`.

## Decision

1. Keep the unqualified `100M / 2B` chat command mapped to the stable pretrained artifact.
2. Add an explicit `--sft` selector that maps `100M / 2B` to `100m-2b-sft-s0-001`.
3. Use `trainer.state.load_trainer_state_file` in `chat.py` so inference accepts both historical plain-pickle trainer states and streamed `torch.save` trainer states.
4. Preserve completed-checkpoint verification and SFT identity verification before inference.

## Consequences

The canonical completed-SFT chat command is:

```bash
python chat.py --model_params 100M --num_tokens 2B --sft
```

The pretrained baseline remains available with:

```bash
python chat.py --model_params 100M --num_tokens 2B
```

The two stages are therefore explicit, and streamed DDP SFT checkpoints do not require a conversion step before interactive inference.
