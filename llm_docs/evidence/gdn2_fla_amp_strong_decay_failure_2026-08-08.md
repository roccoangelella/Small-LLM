# FLA GDN-2 trainer-AMP strong-decay failure — 2026-08-08

## Context

After standalone FLA GDN-2 operator qualification and an initial full-layer integration probe, the 20M/500M launcher was wired to resume the verified `step-00004000` checkpoint with the FLA CUDA backend.

The first real resume attempt restored `step-00004000` successfully and entered global steps `4001-15264`, but the trainer hit a Triton dtype compilation error before update 4001 completed. The error was `Both operands must be same dtype. Got fp32 and fp16` inside FLA GDN-2 WY recomputation. No new optimizer update was committed, so `step-00004000` remains the valid checkpoint.

The adapter was then changed to canonicalize FLA compute tensors (`q`, `k`, `v`, erase, write) to FP16 while leaving log-decay and recurrent state FP32.

## AMP-realistic integration probe

The integration probe was strengthened to match the trainer contract: FP32 model parameters with CUDA FP16 autocast, saved `gdn_chunk_size=32`, and FLA fixed runtime chunk 64.

User-reported output:

```text
[layer] normal_decay_amp  fp32_params+fp16_autocast  saved_config_chunk=32  FLA_runtime_chunk=64
    layer output                       PASS
    grad x                             PASS
    grad A_log                         PASS
    grad dt_bias                       PASS
    grad q_proj.weight                 PASS
    grad k_proj.weight                 PASS
    grad v_proj.weight                 PASS
    grad q_conv.weight                 PASS
    grad k_conv.weight                 PASS
    grad v_conv.weight                 PASS
    grad erase_proj.weight             PASS
    grad write_proj.weight             PASS
    grad decay_proj.0.weight           PASS
    grad decay_proj.1.weight           PASS
    grad output_gate.0.weight          PASS
    grad output_gate.1.weight          PASS
    grad output_gate.1.bias            PASS
    grad output_norm.weight            PASS
    grad out_proj.weight               PASS

[layer] strong_decay_-6_amp  fp32_params+fp16_autocast  saved_config_chunk=32  FLA_runtime_chunk=64
    layer output                       PASS
    FAIL: AssertionError: Tensor-likes are not close!

Mismatched elements: 9216 / 16384 (56.2%)
Greatest absolute difference: nan at index (0, 0, 0)
Greatest relative difference: nan at index (0, 0, 0)

SUMMARY
layer_forward_backward_parity: False
checkpoint_parity: None
trainer_amp_contract_tested: True
NOT QUALIFIED
```

Because the first gradient checked after the layer output is the input gradient, the failure occurs at or before `grad x`. The probe version did not yet separately report which side (adaptive reference or FLA) contained the NaNs.

## Important reinterpretation of prior evidence

The earlier standalone FLA strong-decay `forward+backward` benchmark only timed `.backward()` and did not assert that stress-case gradients were finite or matched the recurrent/adaptive oracle. Therefore this AMP strong-decay gradient failure does not contradict the earlier standalone benchmark; it closes a previously acknowledged qualification gap.

The standalone probe used `disable_recompute=True`. The integrated backend currently uses the default `disable_recompute=False`. In FLA v0.5.1, `disable_recompute=False` causes backward to recompute WY/state intermediates before the gradient kernels, while `True` retains and reuses the forward intermediates. This is now the next isolated hypothesis to test.

## Current safety state

- Stop/avoid the active long 500M resume attempt while this correctness gate is red.
- `step-00004000` remains the valid verified checkpoint because update 4001 did not complete in the failed resume attempt.
- Do not authorize resumed training with FLA until strong-decay gradients are finite and match the adaptive reference under the real FP32-parameter + FP16-autocast contract.
- A focused probe was added at `kaggle/run_gdn2_fla_strong_decay_amp_probe.py` to test `disable_recompute=True` only and explicitly report non-finite gradients on reference vs FLA sides.
