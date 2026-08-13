# 100M/2B SFT step-250 NCCL cadence timeout — 2026-08-13

The second live 100M/2B SFT attempt used two Tesla T4s, per-rank execution
microbatch 2, FP16, PyTorch 2.10.0, Triton 3.6.0, and `fla-core==0.5.2`.
Microbatch 2 passed the no-step backward prewarm that had OOMed at microbatch 4
and completed 250 real optimizer updates.

The final completed update reported:

```text
step: 250
block: 249
consumed loss-bearing targets: 8,044,261
loss: 2.110764980316162
gradient norm: 0.8773994445800781
gradient clipped: false
loss scale: 32768
overflow events total: 1
peak allocated bytes: 8,345,723,392
peak reserved bytes: 11,695,816,704
targets/second: 3,690.0396413788617
```

This establishes that microbatch 2 is finite and has material T4 memory
headroom for ordinary training. The failure occurred after the update at the
first 250-step evaluation/checkpoint/publication boundary.

Rank 0 began the real 16-block validation and 16-case greedy behavior suites.
Rank 1 used the intended dummy rank-zero-only side effects, advanced to the
next session step, and entered the default NCCL barrier. Rank 0 did not enter
that barrier within NCCL's ten-minute watchdog window. Rank 1 reported:

```text
WorkNCCL(SeqNum=4012, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
Timeout(ms)=600000) ran for 600061 milliseconds before timing out
```

The watchdog and `torchrun` then terminated both workers. There was no CUDA OOM
report. The one-element all-reduce and exact timeout match the next-step NCCL
barrier, not an optimizer collective from step 250.

The training loop orders evaluation before local checkpoint save and remote
publication. No `sft_evaluation`, `local_checkpoint`, or `remote_publication`
event appeared after the step-250 record. Therefore the 250 completed updates
are not an exact-resume point and the next attempt must start fresh from the
verified parent.

Implementation commit `2e8fcdf8ed57ec6b998ac1d915ce161f79bfa8ef`
creates a separate Gloo control group with a one-hour timeout and routes SFT
prewarm, next-step, and final rank rendezvous through it. NCCL remains the DDP
training backend, but rank 1 no longer holds an NCCL collective open while
rank 0 performs long cadence side effects. This implementation is covered by
focused unit tests but remains pending the next live step-250 cadence boundary.
