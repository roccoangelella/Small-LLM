# R-SFT R0 Qualification Runbook

Run the production S0-versus-R-SFT evaluator through the Kaggle launcher:

```bash
python kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite full
```

The report retains:

- `eval_core_v1` base retention;
- S0 validation/test loss;
- R-SFT production-bundle validation/test loss;
- deterministic instruction-behavior probes with final-answer extraction;
- native-budget base qualitative regressions;
- novel mechanically scored reasoning probes;
- repeated reasoning sampling;
- atomic reasoning-protocol telemetry.

General qualitative decoding uses:

```text
greedy:  temperature=0, top_p=1, top_k=0
sampled: temperature=1, top_p=1, top_k=0
budget:  native per prompt
```

Reasoning pass@1 remains a separate task-specific sampling protocol and is not
the general model-generation contract.

Interpret reasoning acquisition, protocol health, instruction behavior and
base/S0 retention together; there is no single master score.
