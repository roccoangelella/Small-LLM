# Pretraining Evaluation Targets

_Last updated: 2026-08-05_

## Why this document exists

The approximately-20M model has now completed its first 10M-token pretraining run. We need targets that are useful for the next 100M- and 1B-token runs without pretending that perplexities copied from unrelated papers are directly comparable.

The main conclusion is simple:

- GPT-2 is a valid historical **base-model** reference because the released family was trained with autoregressive next-token pretraining and evaluated zero-shot, without instruction tuning or RLHF.
- Published GPT-2 perplexities are **not** direct score targets for this project. Perplexity changes with the corpus, tokenizer, context construction, document boundaries, and evaluation preprocessing.
- The useful project targets are therefore same-model, same-tokenizer, same-held-out-distribution targets. External suites such as BabyLM, LAMBADA, BLiMP, and EWoK should be added as separate capability measurements.

## What “GPT-2 was only pretrained” means

The original GPT-2 release used a simple autoregressive objective: predict the next token from previous tokens in WebText. Its headline downstream results were zero-shot and the paper explicitly says that no training or fine-tuning was performed for the language-modeling benchmark results.

That makes GPT-2 closer to our current checkpoint than an instruction-tuned chat model. It does not make the comparison automatically fair:

- GPT-2 Small has about 117M paper-counted parameters, versus our 20,637,592 parameters.
- GPT-2 was trained on a much larger WebText corpus.
- The GPT-2 paper used dataset-specific invertible de-tokenizers for several reported perplexities and observed changes of 2.5–5 perplexity points from that preprocessing alone.
- WikiText, PTB, LAMBADA, WebText, and ClimbMix have different intrinsic difficulty and domain mixtures.

The published GPT-2 Small zero-shot values remain useful historical reference points, for example LAMBADA perplexity 35.13 with 45.99% accuracy and WikiText-2 perplexity 29.41. They are not pass/fail thresholds for our ClimbMix validation loss.

A genuinely fair architecture comparison must keep fixed, as far as possible:

```text
tokenizer and vocabulary
evaluation documents and target tokens
context length and document-boundary policy
parameter budget
training-token exposure
optimizer and schedule budget
data ordering or at least data distribution
scoring implementation
```

The cleanest project baseline remains the parameter-matched Plan C transformer trained on the same data. Evaluating the released GPT-2 Small checkpoint on a frozen project evaluation set is still worthwhile as a historical capability reference, especially because the project already uses GPT-2 token IDs, but it must be labelled as a larger model trained on far more data.

## Current project anchor

The accepted 10M-token run produced:

```text
parameters: 20,637,592
accepted source tokens: 10,000,662
planned target tokens: 10,006,528
final validation loss: 6.136690 nats/token
final validation perplexity: 462.520157
```

At the 50-update point, after 1,638,400 target tokens, validation loss was 7.915478 and perplexity was 2,739.35. The full run therefore gives us one real project learning-curve segment instead of requiring a forecast from unrelated models alone.

Data exposure relative to model size is:

```text
10M tokens:   0.485 tokens/parameter
100M tokens:  4.846 tokens/parameter
1B tokens:   48.455 tokens/parameter
```

The original Chinchilla compute-optimal projection is roughly 20 training tokens per parameter around the model sizes it studied. Applied only as a rough orientation, that puts a 20.64M-parameter model near 413M tokens. This is not an exact optimum for our architecture, optimizer, data, hardware, or inference-amortization goal, but it shows why 10M is extremely data-starved, 100M is still undertrained, and 1B is a reasonable long-run learning experiment rather than an absurd amount of data.

## Planning loss and perplexity ranges

These ranges are **forecast bands**, not published apples-to-apples records and not hard authorization gates.

Two estimates bracket the forecast:

1. The data-limited scaling relation in Kaplan et al. uses an exponent near `0.095`. Calibrated from our measured 10M loss, it predicts approximately `4.93` loss at 100M tokens and `3.96` at 1B tokens.
2. Fitting only our observed 1.638M-to-10M validation trajectory gives a steeper exponent near `0.141`. Continuing that trajectory predicts approximately `4.44` at 100M and `3.21` at 1B. This is optimistic because it is based on only two points and includes the early-training transient.

The practical band is the interval between those two estimates, rounded outward:

| Seen target tokens | Working same-corpus range | Perplexity range | Stretch result | Investigate if |
|---:|---:|---:|---:|---:|
| 10M | `6.0–6.3` | `403–545` | `< 6.0` | `> 6.4` |
| 100M | `4.4–5.0` | `81–148` | `< 4.4` | `> 5.2` |
| 1B | `3.2–4.0` | `25–55` | `< 3.2` | `> 4.2` |

The finished 10M run, at `6.136690 / 462.52`, lands inside the working 10M band. It should be treated as the measured project anchor. The 10M range is centered on that result and is less independently grounded than the 100M and 1B bands.

Interpretation:

- A 100M result around loss `4.9` would be consistent with conservative data scaling.
- A 100M result around `4.4–4.6` would indicate that the strong early project trajectory mostly continued.
- A 1B result around `3.9–4.0` would still be credible progress.
- A 1B result around `3.2–3.4` would be very strong for this trajectory.
- Crossing a warning value does not by itself prove a bad architecture. First check the fixed evaluation identity, schedule, data mixture, optimizer state, and whether the run actually consumed the intended tokens.

These targets are valid only if future checkpoints are evaluated on a common frozen distribution. Rebuilding a tiny validation split independently for every token budget can move the score enough to confuse data scaling with evaluation-sample drift.

## External low-data references

### BabyLM

BabyLM is the closest public evaluation programme to our logarithmic token budgets. The 2025 challenge required checkpoints at every 1M words through 10M, every 10M through 100M, and every 100M through 1B, and evaluated intermediate checkpoints on a fast zero-shot suite.

Its GPT-2 baseline is not numerically identical to our setup: it uses the GPT-2 Small architecture, a 16k tokenizer, 512-token training chunks, different data, and repeated epochs. Still, it gives useful capability targets once we run the same evaluation pipeline.

For the 2025 Strict-Small GPT-2 baseline after about 100M word exposures, the model card reports:

```text
BLiMP accuracy:             66.36
BLiMP Supplement accuracy:  57.07
EWoK accuracy:              49.90
Entity Tracking accuracy:   13.90
WUG accuracy:               52.50
```

The challenge aggregate table reports the GPT-2 baseline at:

```text
about 100M word exposures: NLP score 49.1, macro average 34.5
about 1B word exposures:   NLP score 55.4, macro average 36.2
```

These are external yardsticks, not expected scores. Tokens are not words; the model and data are larger/different; several task scores are near chance at low exposure; and fine-tuned task results measure representation transfer rather than raw zero-shot behavior.

BabyLM nevertheless provides a good evaluation design: measure the whole learning trajectory, use fast zero-shot tasks at intermediate checkpoints, and reserve fine-tuned probes for important final checkpoints.

### Recent under-1B base-model practice

The February 2026 Qwen3.5-0.8B-Base release is explicitly a pre-trained-only checkpoint. Its language stack uses six repetitions of three Gated DeltaNet layers followed by one gated-attention layer. That recent under-1B design is architecturally close enough to reinforce two project choices:

- hybrid recurrent/linear attention plus periodic full attention remains a current design, not only a historical curiosity;
- evaluation should include capability and efficiency measurements, because perplexity alone cannot tell us whether the hybrid is buying useful long-context behavior or better serving characteristics.

Qwen3.5 is not a numerical target for this experiment: it is roughly forty times larger than the smoke model, multimodal, uses a much larger vocabulary and context, and was trained with a vastly different data budget.

## Metrics beyond perplexity

Perplexity is the exponentiated average cross-entropy. It is convenient, but the underlying loss in nats/token should remain the primary optimization metric because loss differences are easier to average and model.

### Intrinsic predictive quality

Record at minimum:

- validation negative log-likelihood in nats/token;
- perplexity;
- bits per byte, which is more comparable across tokenizers than token perplexity;
- token-weighted and macro-average loss by retained ClimbMix cluster;
- worst-cluster loss;
- top-1, top-5, and top-10 next-token accuracy;
- loss by token-frequency bucket, sequence position, and document-length bucket;
- calibration, such as confidence-versus-correctness ECE or Brier score.

Per-cluster and byte-normalized measurements are especially important here. A lower global perplexity can hide a model that improved strongly on an easy/high-weight domain while becoming worse on rarer domains.

### Sample and compute efficiency

For architecture comparisons, report the curve rather than only the final point:

- loss versus seen target tokens;
- tokens required to cross fixed loss thresholds;
- area under the loss-versus-log-tokens curve;
- fitted local scaling exponent, with its fitted interval and checkpoint range;
- training tokens/second, wall-clock time, peak allocated and reserved VRAM;
- estimated training FLOPs when the implementation makes this trustworthy;
- quality at fixed tokens and quality at fixed wall-clock or compute.

This is more informative than asking only which model won at the last checkpoint. A model that reaches the same loss in half the tokens is a meaningful result even if both eventually converge to a similar score.

### Base-model capabilities

Keep zero-shot and fine-tuned results separate.

Recommended zero-shot or likelihood-scored measurements:

- LAMBADA accuracy and perplexity for long-range final-word prediction;
- BLiMP and BLiMP Supplement for grammatical minimal pairs;
- EWoK for basic world and discourse knowledge;
- BabyLM entity tracking and WUG morphology tasks;
- a small fixed `lm-evaluation-harness` set such as HellaSwag, PIQA, WinoGrande, and ARC Easy/Challenge, reported with exact harness version, prompt template, shot count, and normalization.

At 10M and 100M tokens, many broad knowledge and reasoning scores may remain near chance. That is still useful: the learning curve and the first point at which a task rises reliably above chance are more informative than a single aggregate.

Fine-tuned (Super)GLUE-style probes can be added at major checkpoints to measure whether useful representations are present. They must not be mixed into a “base zero-shot” average.

### Generation quality

Use a fixed blind prompt suite and record both samples and mechanical failure rates:

- repeated 4-gram rate and loop frequency;
- distinct n-grams;
- EOS/termination behavior and completion length;
- malformed UTF-8 or invalid-token events, which should remain zero;
- human ratings for local grammar, coherence, topic continuity, and prompt relevance;
- optional distributional generation metrics such as MAUVE only when enough samples are available and the reference distribution is well defined.

### Robustness, memorization, and efficiency at inference

Important later-stage measurements include:

- exact or near-exact training-string reproduction and canary extraction;
- benchmark contamination checks;
- sensitivity to prompt wording and context truncation;
- prefill and decode throughput;
- first-token and per-token latency;
- peak inference memory;
- KV-cache or recurrent-state bytes per generated token/session;
- segmented, cached, and tokenwise decoding parity.

The last item is particularly important for GDN-2: state efficiency is part of the reason to use the architecture and should be measured rather than assumed.

## Recommended project scorecard

At every logarithmic checkpoint (`10M`, `100M`, `1B`, and later), save one versioned result bundle containing:

```text
fixed-eval loss, perplexity, and bits per byte
cluster-weighted, cluster-macro, and worst-cluster loss
next-token top-1/top-5/top-10 accuracy and calibration
loss-versus-token curve and local scaling fit
BabyLM fast zero-shot suite
LAMBADA plus a small frozen lm-eval task set
fixed generation-suite samples and degeneration metrics
training throughput, wall time, and peak VRAM
inference throughput, latency, and state/cache memory when available
exact model, dataset, tokenizer, harness, and evaluation-set identities
```

For major architecture decisions, run the same bundle for the hybrid and parameter-matched transformer baseline at matched token and compute budgets.

## Evaluation-set work still needed

The current deterministic validation split is enough to show that training improved, but the completed 10M dataset contained only about 10k validation target tokens. That is too small for stable per-cluster reporting and detailed slicing.

Before treating the 100M and 1B target bands as scientific comparison gates:

1. preserve the existing validation result for continuity;
2. create a static `eval_core_v1` from documents permanently excluded from every training budget;
3. keep a small fast subset for frequent checkpoint evaluation and a larger full set for final checkpoints;
4. stratify the full set across retained clusters and retain both macro and mixture-weighted scores;
5. version and hash the exact token arrays, provenance, scoring code, harness versions, and task prompts;
6. report confidence intervals or bootstrap intervals for task accuracy and sliced losses.

The exact size of `eval_core_v1` is not frozen by this document. It should be selected after measuring T4 evaluation time and the minimum per-cluster sample needed for useful uncertainty bounds.

## Sources used for this planning baseline

- OpenAI, *Better language models and their implications* and *Language Models are Unsupervised Multitask Learners* (2019): https://openai.com/index/better-language-models/ and https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020): https://arxiv.org/abs/2001.08361
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022): https://arxiv.org/abs/2203.15556
- Charpentier et al., *Findings of the Third BabyLM Challenge* (2025): https://aclanthology.org/2025.babylm-main.28/
- BabyLM 2025 evaluation pipeline: https://github.com/babylm/evaluation-pipeline-2025
- BabyLM GPT-2 baselines: https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt2 and https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt2
- Qwen3.5-0.8B-Base model card (2026): https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- EleutherAI language-model evaluation harness: https://github.com/EleutherAI/lm-evaluation-harness
