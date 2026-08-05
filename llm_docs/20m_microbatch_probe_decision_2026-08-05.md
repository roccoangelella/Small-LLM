# 20M/100M Microbatch Probe Decision — 2026-08-05

Correction: the user did not cancel the proposed probes for microbatch sizes 6, 8, and 10. The prior interpretation of “end the microbatch probes task” as cancellation was incorrect.

Current decision:

- retain the request to extend the fresh-run microbatch qualification candidates to 1, 4, 6, 8, and 10;
- preserve one immutable 16-sequence block per optimizer update, approximately 32,768 target tokens at context length 2,048;
- changing `microbatch_size` only changes forward/backward slicing within that fixed block and does not expand the optimizer batch;
- do not change the microbatch configuration of an already checkpointed run;
- candidate OOM or numerical failure should reject that candidate rather than invalidate the whole launch.

The optimizer batch remains fixed unless `sequences_per_block` is changed by a separate explicit decision.
