We finally moved from studying and designing to actually qualifying the system.

The final small model is roughly 20.6M parameters and uses the hybrid block architecture we selected: three GDN-2 layers followed by one full gated attention layer, repeated through the network. We also fixed the T4-safe implementation details: context length 2048, FP16, chunkwise GDN-2 with chunk size 32, normal initialization, and seed 17.

For the optimizer, we decided to use a hybrid Muon + AdamW setup. Muon handles ordinary two-dimensional feature transformation matrices, while AdamW handles embeddings, normalization parameters, biases, recurrent dynamics, filters, and the remaining structured parameters. We also added telemetry to measure gradient norms, clipping, optimizer direction RMS, effective update RMS, update-to-weight ratios, memory usage, overflows, throughput, and other useful training signals.

The biggest practical step was building the real finite qualification dataset. We streamed Nemotron-ClimbMix from the pinned source revision and produced a dataset with:

- 10,000,662 accepted source tokens
- context length 2048
- 16 sequences per training block
- 6 training shards and 1 validation shard
- 4,886 training sequences
- 306 training blocks, therefore 306 optimizer updates

The dataset was uploaded durably to Google Drive while it was being built. We then verified local file sizes, SHA-256 checksums, block continuity, source-token attribution, and the exact correspondence between the local and Drive manifests.

We discovered that the old `--full-scan` command did not literally scan every token for schema-v2 shards, so we added a dedicated fail-closed qualification verifier. Its focused tests passed, and the real scan decoded all 10,021,659 stored tokens. No token was outside the GPT-2 vocabulary range and no structural problem was found.

Finally, we generated the exact one-pass training plan from the finished manifest instead of relying on estimates:

- 16 warmup updates
- 228 stable updates
- 62 decay updates
- 306 total updates
- 10,006,528 total training target tokens
- 1 validation block

So the dataset side is now actually done and accepted. The next step is Kaggle: package the dataset privately, run the complete test suite and T4 harness on one exact commit, then execute a 20-update W&B preflight. Only after checking stability, memory, overflow, clipping, and optimizer telemetry will we authorize the complete 306-update run.

We are finally very close to seeing whether the whole thing really trains.