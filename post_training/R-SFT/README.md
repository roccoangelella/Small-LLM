# R-SFT R0 data lane

This folder owns the first reasoning-SFT dataset path: Gemini prompt generation, strict JSON parsing, the R0 skill/difficulty matrix, matched delimiter serialization, S0 instruction retention, R-SFT tokenization, and immutable bundles consumed by the Kaggle trainer.

## First 630-example pilot

The frozen pilot is 7 skills x 3 difficulty bands x 30 examples = 630 Gemini-generated examples. At the default batch size of 10 this is 63 API calls.

Dry-run the full plan without credentials or API calls:

```bash
python post_training/R-SFT/produce.py pilot \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --token-spec /path/to/reasoning-tokens.json \
  --output-dir artifacts/rsft-r0-pilot-630 \
  --dry-run
```

Run generation and build both matched bundles:

```bash
python post_training/R-SFT/produce.py pilot \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --token-spec /path/to/reasoning-tokens.json \
  --output-dir artifacts/rsft-r0-pilot-630
```

`GEMR_API_KEY` and `LLM_ENDPOINT` are resolved by `dataset.py` from the environment or repository `.env`. Every successful Gemini batch is saved before the next request. Re-running the same command skips valid completed batches, so an interrupted 63-call run resumes rather than restarting.

Generation can also be run separately before the delimiter token spelling is frozen:

```bash
python post_training/R-SFT/produce.py generate \
  --output-dir artifacts/rsft-r0-pilot-630/generation
```

Then build later with:

```bash
python post_training/R-SFT/produce.py build \
  --reasoning-jsonl artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --token-spec /path/to/reasoning-tokens.json \
  --output-dir artifacts/rsft-r0-pilot-630/bundles
```

## Reasoning token spec

The three atomic IDs are already fixed at 50257, 50258, and 50259, but their text spellings are intentionally not hardcoded by dataset production. Supply either the full tokenizer metadata or a compact JSON object:

```json
{
  "reasoning_start": "<chosen reasoning-start spelling>",
  "reasoning_end": "<chosen reasoning-end spelling>",
  "answer_start": "<chosen answer-start spelling>"
}
```

The builder validates the mapping and writes a full `reasoning-tokens.json` into both arm bundles. The textual arm uses ordinary GPT-2-tokenized `Reasoning:` / `Answer:` boundaries and never emits IDs 50257-50259; it still carries the same token spec because both ablation arms use the same promoted 50,260-vocabulary model architecture.

## Output layout

The one-shot pilot produces:

```text
artifacts/rsft-r0-pilot-630/
  generation/
    batches/                  # resumable schema-valid Gemini responses
    reasoning.jsonl           # frozen 630-example corpus
    generation-manifest.json
  bundles/
    source-manifest.json      # shared reasoning split + S0 retention identity
    reasoning.jsonl
    pilot-manifest.json
    atomic/
      bundle-manifest.json
      reasoning-tokens.json
      source-manifest.json
      train/
      validation/
      test/
    textual/
      bundle-manifest.json
      reasoning-tokens.json
      source-manifest.json
      train/
      validation/
      test/
```

The 30 examples in every skill x difficulty cell are deterministically partitioned as 28 train, 1 validation, and 1 test example, so both held-out splits cover all 21 cells. The exact same partition is used by both delimiter arms.

The 10% retention lane is sampled only from the instruction sources in the completed S0 training bundle. It preserves the S0 bundle's recorded instruction source shares and reuses the exact stored S0 token IDs/target masks; ClimbMix replay is not part of R-SFT retention. Both ablation arms receive the identical selected retention record IDs in the identical semantic record order.

Because textual delimiters occupy a different number of tokens from three atomic delimiters, one identical retention sample cannot be exactly 10.000% of both arm token totals simultaneously. The builder therefore chooses one symmetric retention target from the mean atomic/textual reasoning-token totals, reuses the same retained records in both arms, and records each arm's realized retention share in `pilot-manifest.json`.

Verify an already-built matched root with:

```bash
python post_training/R-SFT/produce.py verify \
  --dataset-dir artifacts/rsft-r0-pilot-630/bundles
```

The verifier delegates each arm to the existing native SFT `verify_bundle()` contract used by the trainer.
