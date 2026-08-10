# 20M/100M Microbatch Probe Decision — 2026-08-05

The user requested that the expanded 1/4/6/8/10 probe implementation be rolled back after the Kaggle run reached microbatch 10 OOM rejection, selected microbatch 4, and then stalled during W&B initialization timeout.

Current operational decision:

- restore the repository code to the state before the expanded microbatch runtime was introduced;
- use the original first-session qualification of microbatch 1 versus microbatch 4;
- keep actual training fixed at `microbatch_size=4` when that gate passes;
- preserve one immutable 16-sequence block per optimizer update, approximately 32,768 target tokens at context length 2,048;
- do not use the 6, 8, or 10 probes in the current launcher;
- preserve the informative console formatter added before the expanded-probe implementation;
- do not interpret this rollback as changing `sequences_per_block` or expanding the optimizer batch.

The rollback restores code behavior equivalent to repository commit `f116890ab60462e43398fdc87943f0fc0ed1ec8a`, apart from this decision record.
