---
status: evidence
observed_at: 2026-09-08
artifact: topk15-base-prompt-evidence
sources:
  - 100m_2b_sft_v2_augmented.json
  - 100m_2b_sft_v2_augmented_gemrouter.json
  - 100m_10b_sft_v2_augmented.json
  - 100m_10b_sft_v2_augmented_gemrouter.json
---

# Top-k15 Base Prompt evidence

New diagnostic sampling contract: `temperature=1.0`, `top_p=1.0`, `top_k=15`, `seed=17`. Existing greedy evidence is repeated compactly beside the new higher-temperature view to make future comparisons self-contained. Only `prompt` and shortened `continuation` text are preserved from model answers; full JSON contents are intentionally omitted.

## GemRouter semantic-judge summaries

| Model state | top-k15 accuracy | correct / 100 |
| --- | ---: | ---: |
| 100M / 2B pretrained parent | 5% | 5 |
| 100M / 2B SFT | 5% | 5 |
| 100M / 10B pretrained parent | 7% | 7 |
| 100M / 10B SFT | 6% | 6 |

### 100M / 2B pretrained parent — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: France

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 is multiplied by 8.

prompt: Order number: 4187; status: shipped.\nOrder number:
continuation: 4187; status: shipped. …

### 100M / 2B pretrained parent — sampled top-k15

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 divided by 8 equals 2.1.

prompt: Order number: 4187; status: shipped.\nOrder number:
continuation: 4187\n\nQuestion: Is the text the only one of its kind? …

### 100M / 2B SFT — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 multiplied by 8 is 7 multiplied by 8.

prompt: Order number: 4187; status: shipped.\nOrder number:
continuation: 4187\n\nQuestion: What is the total number of letters in the text? …

### 100M / 2B SFT — sampled top-k15

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 divided by 8 equals 2.75.

prompt: Item: cat\nClassify as animal or object:
continuation: animal, object, or both.\nAnswer: animal\nExplanation: …

### 100M / 10B pretrained parent — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris\n\nQuestion: What is the name of the book that explores the history of the French Revolution? …

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 is multiplied by 8.

prompt: Item: cat\nClassify as animal or object:
continuation: animal\n\nQuestion: What is the main difference between a dog and a cat …

### 100M / 10B pretrained parent — sampled top-k15

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 divided by 8 equals 8. …

prompt: Item: cat\nClassify as animal or object:
continuation: animal\nClassify as animal, animal, person, thing: …

### 100M / 10B SFT — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris\n\nQuestion: What is the main reason why the city is famous for its architecture? …

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 * 8 = 81.

prompt: Order number: 4187; status: shipped.\nOrder number:
continuation: 4187; status: shipped. …

### 100M / 10B SFT — sampled top-k15

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 divided by 8 equals 100.

prompt: Order number: 4187; status: shipped.\nOrder number:
continuation: 4187\n\nQuestion: Is an air-sealed seal always required by the seal? …

## Correct top-k15 objective cases

- 2B parent: `capital_france`, `water_formula`, `order_number`, `username`, `batch_code`.
- 2B SFT: `capital_france`, `hardest_natural_substance`, `order_number`, `cat_animal`, `python_programming`.
- 10B parent: `capital_france`, `earth_star`, `order_number`, `tracking_code`, `cat_animal`, `dolphin_mammal`, `python_programming`.
- 10B SFT: `capital_france`, `capital_italy`, `capital_spain`, `order_number`, `tracking_code`, `room`.
