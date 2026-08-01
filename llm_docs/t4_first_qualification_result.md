# First T4 GDN-2 Qualification Result

_Last updated: 2026-08-01_

## Project conclusion

The first Kaggle NVIDIA Tesla T4 qualification run establishes that the current Small LLM GDN-2 model can execute on the target T4. The earlier uncertainty about whether the architecture could run at all on compute capability 7.5 is resolved.

This is an execution-feasibility result, not yet a pretraining authorization.

## Evidence

Using the approximately-20M smoke model at context 2,048 and microbatch 1:

- the PyTorch chunkwise GDN-2 hybrid completed FP32 training steps with chunk sizes 16, 32, and 64;
- it completed FP16 training steps with chunk sizes 16 and 32;
- losses decreased, gradients remained finite, and no FP16 scaler reductions were observed in those successful short runs;
- peak allocated memory remained below approximately 2.8 GiB, so smoke-model memory capacity is not the immediate T4 blocker;
- FP16 chunk size 64 produced non-finite chunkwise values and failed;
- the Plan-B SWA hybrid completed successfully and remains the operational fallback.

## Remaining blocker

The chunkwise implementation is intended to be a parallel evaluation of the same GDN-2 recurrence as the tokenwise oracle. The T4 parity tests did not pass:

- FP32 chunkwise outputs or states exceeded the strict recurrent-reference tolerances;
- FP16 parity cases produced non-finite values for all tested chunk sizes.

Therefore the current conclusion is:

> GDN-2 execution on the T4 is viable, but the chunkwise-versus-tokenwise parity defect must be diagnosed and fixed before the chunkwise backend is trusted for real pretraining.

After the correctness issue is fixed, the qualification must be repeated. Throughput also remains an operational concern because the ordinary-PyTorch GDN-2 backend was substantially slower than Plan B in this short run.

## Decision boundary

Do not replace the primary architecture solely from this first report. Keep GDN-2 as the intended architecture while investigating parity and FP16 stability. Plan B remains available if a corrected GDN-2 backend cannot achieve acceptable correctness and throughput on the T4.
