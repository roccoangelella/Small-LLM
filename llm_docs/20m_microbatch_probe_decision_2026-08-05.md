# 20M/100M Microbatch Probe Decision — 2026-08-05

The user ended the proposal to expand the first-session microbatch probe to candidates 6, 8, and 10.

Frozen operational interpretation:

- keep the current experiment's optimizer batch at one immutable 16-sequence block per optimizer update;
- this is approximately 32,768 target tokens per optimizer update at context length 2,048;
- `microbatch_size=4` only splits that block into four forward/backward slices before one optimizer step;
- do not add probes for microbatch sizes 6, 8, or 10 to the current 20M-model/100M-token run;
- do not change an existing checkpointed run's microbatch size.

Therefore, the optimizer batch is not being expanded. Changing `sequences_per_block` would be the operation that changes the effective optimizer batch and requires a separate explicit decision.
