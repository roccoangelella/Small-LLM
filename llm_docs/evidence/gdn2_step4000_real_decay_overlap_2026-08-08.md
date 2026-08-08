---
status: evidence
observed: 2026-08-08
---

# Step-4000 real-data GDN-2 decay overlaps FLA v0.5.1 AMP backward failure region

The user ran the forward-only real-checkpoint telemetry probe against the verified 20M/500M `step-00004000` checkpoint and train block 4000.

User-reported summary:

```text
# Small-LLM step-4000 real-data GDN-2 decay telemetry

# SUMMARY

any_individual_g_le_minus_0.75: True
any_64tok_mean_g_le_minus_0.75: True
real checkpoint overlaps the tested FLA failure region; do not resume chunk-GDN2 training
JSON report: /kaggle/working/gdn2_step4000_decay_telemetry.json
```

Context from the immediately preceding trainer-AMP forced-decay sweep:

```text
passing: [-0.25, -0.5]
failing: [-0.75, -1.0, -1.25, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0]
first failing tested point: g=-0.75
```

Interpretation:

- The synthetic backward failure regime is not merely an extreme artificial `g=-6` case.
- The actual trained step-4000 checkpoint produces individual log-decay values at or below `-0.75` on the next real training microbatch.
- More importantly, at least one real 64-token region has mean log-decay at or below `-0.75`, directly overlapping the tested constant-decay region where FLA v0.5.1 chunk backward produced non-finite/incorrect gradients under the trainer's FP32-parameter + FP16-autocast contract.
- Therefore FLA v0.5.1 `chunk_gdn2` is not qualified for resumed training of the active 500M trajectory.
- The verified step-4000 checkpoint remains the latest accepted trajectory point; no FLA attempt has committed update 4001.

This evidence does not by itself authorize clipping/bounding learned decay. The next exact-semantics optimization candidate should be separately qualified before changing model behavior.
