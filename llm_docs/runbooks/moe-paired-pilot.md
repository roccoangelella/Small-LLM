# Bounded MoE pilot

Protocol: [ADR 0168](../decisions/0168-moe-paired-pilot-controller-and-observation.md).
The local code is prepared for D100/M0/M1 on the existing Beam RTX4090 and Modal H100 resources. Cloud execution remains unqualified until the exact image passes CUDA/FLA and volume-resume checks. A command below is not authorization to spend or publish.

The pilot and `--experiment-dir` observation are single-process only. Do not enable observation in the dual-T4/DDP entrypoints: rank ownership and shared artifact writes are not implemented for distributed observation.

## Local verification

From the repository root, with the model dependencies already installed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
python -m unittest -v \
  tests.test_moe_model tests.test_moe_controller tests.test_moe_evaluation \
  tests.test_trainer_fp16_overflow tests.test_moe_cli_observation \
  tests.test_probe_compare tests.test_eval_local tests.test_moe_pilot_protocol
```

The CLI test calls real `python -m MOE_model` processes and the real probe-comparison command on temporary schema-v2 data. It compares continuous training with checkpoint/resume, including optimizer, controller and RNG, and checks fixed probes/profile regions. The CUDA test skips without a GPU; provider tests substitute the SDK boundary, not the child training loop.

Before a real pilot, run `tests.test_moe_controller.TestMoEController.test_cuda_precision_forward_backward_step` inside the exact GPU image. Then use a separate qualification namespace for a bounded real-data step/checkpoint/resume on each selected lane. Verify resumed next block, controller state and artifacts. Keep the cap frozen within each namespace; use a new run ID to change a cap or any recipe setting. Do not silently reduce microbatch/precision after an OOM.

## Launch inputs and prepared data

Required inputs: clean source commit; explicit run ID; a complete, verified `modal-2b-b64` schema-v2 dataset already mounted under `/data`; verified `eval_core_v1`; actual provider resources/quote/credit eligibility; space for retained checkpoints; authorization for the bounded cloud action. No new credentials belong in this repository.

`moe_pilot.py` is framework-free at import and performs CPU preflight before dispatch. It freezes a provider/run/arm namespace under `/runs/moe-pilots/`, manifest hash, full dataset-derived WSD schedule and observation policy. The pilot cap does not shorten warmup/decay. Recipes are explicit in the child command: 100M substantive GDN2, normal init, seed 17, hybrid Muon/AdamW, LR 3e-4, gradient clip 1, decay 0.1; GDN chunk 32 for both FP16/BF16. Defaults use two fixed validation sequences, no full LM logits, profiles 50/400/900 filtered by cap and selected checkpoints 0/25/50/100/200/300/500/750/1000 filtered by cap.

After authorization, examples from a clean repository:

```bash
PILOT_SOURCE_SHA=$(git rev-parse HEAD)
python beam/moe_launch.py --arm M0 --run-id moe-pilot-001 \
  --dataset-dir /data/modal-2b-b64-dataset-001 --steps 1000 \
  --precision fp16 --microbatch-size 4 --source-commit "$PILOT_SOURCE_SHA"
python beam/moe_launch.py --arm M1 --gamma 0.001 --run-id moe-pilot-001 \
  --dataset-dir /data/modal-2b-b64-dataset-001 --steps 1000 \
  --precision fp16 --microbatch-size 4 --source-commit "$PILOT_SOURCE_SHA"
modal run modal/moe_launch.py --arm M0 --run-id moe-pilot-001 \
  --dataset-dir /data/modal-2b-b64-dataset-001 --steps 300 \
  --precision fp16 --microbatch-size 16 --source-commit "$PILOT_SOURCE_SHA"
modal run modal/moe_launch.py --arm M1 --gamma 0.001 --run-id moe-pilot-001 \
  --dataset-dir /data/modal-2b-b64-dataset-001 --steps 300 \
  --precision fp16 --microbatch-size 16 --source-commit "$PILOT_SOURCE_SHA"
```

D100 uses `--arm D --steps 100`, no gamma, on the same lane settings. It measures systems behavior; do not rerun a completed dense endpoint. Use exact mounted paths from the preflight, not these example paths without checking. `modal run` itself contacts Modal, including when its local entrypoint receives a dry-run option.

A retry reuses the same command/namespace. The runner verifies the latest complete joint checkpoint and computes only the remaining successful updates. A partial/corrupt latest checkpoint cannot become a completed result. There is no W&B/HF publication in this pilot. Modal commits the immutable checkpoint tree at its event and commits remaining artifacts at finalization; Beam uses its durable volumes. The stdout callback does not pause the child while committing. Preserve the selected checkpoint directories; no latest-only cleanup.

## Intrinsic evaluation and probe analysis

Run on the retained checkpoint snapshot and the same verified corpus. The scorer and cluster definitions remain canonical:

```bash
python -m trainer.eval_local --moe \
  --checkpoint-dir /runs/moe-pilots/beam/moe-pilot-001/M1/checkpoints/step-00000300 \
  --eval-dir /data/eval_core_v1 --split fast \
  --output-json /runs/moe-pilots/beam/moe-pilot-001/M1/eval-step-00000300.json \
  --device cuda --precision fp16 --batch-size 4 --bootstrap-samples 1000
```

Omit `--moe` for D. Evaluate all selected checkpoints for paired curves, with the same eval precision/configuration. The adapter performs no HF download, prompt tier or external judge call. The corpus requires `manifest.json`, both fast/full binary files and their records sidecars. Full verification applies even when selecting fast. Record checkpoint/corpus/source hashes and evaluation config; bootstrap is by document within this frozen corpus, not across training seeds.

Compare fixed probes and optionally interval weight changes within one run:

```bash
python -m trainer.probe_compare \
  /runs/moe-pilots/beam/moe-pilot-001/M1/experiment/step-00000200-probe.pt \
  /runs/moe-pilots/beam/moe-pilot-001/M1/experiment/step-00000300-probe.pt \
  --checkpoint-before /runs/moe-pilots/beam/moe-pilot-001/M1/checkpoints/step-00000200 \
  --checkpoint-after /runs/moe-pilots/beam/moe-pilot-001/M1/checkpoints/step-00000300 \
  --output /runs/moe-pilots/beam/moe-pilot-001/M1/probe-200-300.json
```

For M0/M1 probe comparisons at matched steps, omit checkpoint-delta options: different arm configs intentionally have different joint identities. CE deltas use the target mask; routing metrics count routed positions. Norm samples are post-clipping. Checkpoint Δp is an interval difference, not an individual Muon update.

## Interpretation and closure

The outer profiler region `session_step` includes data wait and acknowledgement; nested `forward_ce`, `backward`, and `optimizer` regions distinguish compute phases. The profile event reports capture plus export time. Keep clean update timing separate from profiled updates, validation, probe and checkpoint overhead. Compare hardware only at matched tokens/LR: H100 300 is still warmup, 4090 reaches 1000. Test observation-on/off from the same snapshot when estimating overhead. Retaining all planned MoE checkpoints across both lanes is about 93.2 GB in the fully allocated state scenario, before dense, profiles and metadata; use measured bytes for storage planning.

Session summaries and runner runtime are diagnostic wall clocks, not bills. Runtime persistence includes handled failed attempts and result verification, but excludes final volume commit/provider startup and can miss abruptly terminated tails. Provider receipts must account for CPU/RAM, all attempts, storage and credit rules; useful-token cost divides total spend by unique durable targets. Function completion ends compute; retain volumes until artifacts are verified and their later cleanup is explicitly chosen.
