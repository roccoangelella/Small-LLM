# MoE step-75k gibberish: wrong tokenizer at chat time — 2026-09-23

The local, manifest-verified checkpoint `moe-100b-superbpe-003-chat-best-step-00075000/step-00075000` reports 9,830,400,000 consumed targets and validation CE 2.725040. `python chat.py --moe` originally loaded `tokenizer/superbpe_8000.json` (SHA-256 `4220ad83…899b06`), selected solely because `semantic_vocab_size == 8000`. Edo's `edo/tokenizer-v2-corpus` branch adds the *different* worker-v2 artifact `tokenizer/superbpe_8000_v2.json` (SHA-256 `068e20ef…d373bb`). Both have 8,000 IDs and EOS 7992, but their text-to-ID mappings differ. The snapshot metadata contains the corpus manifest identity, not a directly verified tokenizer SHA; the experiment below establishes an inference mismatch, not a full audit of all training shards.

On one RTX 4060 Laptop GPU, `model.eval()`, FP16 autocast, the **same English text** and two tokenizer encodings, next-token cross-entropy over the encoded text prefixed by EOS 7992:

| checkpoint | original v1 tokenizer | Edo's v2 tokenizer |
|---|---:|---:|
| `moe-100b-superbpe-001`, step 15,000 | 3.284 (40 IDs) | 10.589 (39 IDs) |
| `moe-100b-superbpe-003` chat snapshot, step 75,000 | 10.569 (40 IDs) | 3.074 (39 IDs) |

Text: `In the morning, the children walked to school together. Their teacher greeted them at the door and asked how they were feeling. The sun was shining, and the classroom was warm and bright.` The token counts differ; each CE is per token in that tokenizer, so use the reversal as a diagnostic, not a matched-token quality ranking. V1 on step 75k gives the reported ` int limlt … higature isason …` greedy output. V2 on the same weights and `The capital of France is` gives ` located in central Europe. The capital is Montpelier, which was founded in 1592. …`; `Once upon a time, in a small village, there lived a` gives an English continuation about a girl named Mia. This recovers words, not factual accuracy or robust long-form behavior. Switching the step-75k v1 experiment from FLA/FP16 to adaptive GDN-2/FP32 changed clean-text CE only 10.569 → 10.566; precision/kernel choice is not the cause.

**Remaining limitations:** The snapshot is pretrained, not instruction-tuned. With v2, the existing `User:\nhello\n\nAssistant:\n` template greedily repeats `Assistant:`. That is a separate prompt/behavior issue, not tokenizer gibberish. The trainer currently accepts 8k shard token IDs without checking the manifest's tokenizer SHA; a future train/validation mix-up would still be possible. Check actual shard manifests before claiming the whole run's corpus identity. Do not substitute v2 globally: run 001 demonstrably needs v1.
