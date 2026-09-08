---
status: evidence
observed_at: 2026-09-08
artifact: compact-eval-answer-samples
sources:
  - 100m_2b_pretraining_v2.json
  - 100m_2b_sft_v2.json
  - 100m_10b_pretraining_v2.json
  - 100m_10b_sft_v2.json
---

# Evaluation-v2 compact answer samples

This record preserves short `prompt` / `continuation` pairs from the four user-supplied evaluation-v2 result JSONs. It intentionally omits token IDs, scoring internals, and other large JSON fields. Samples cover both greedy and higher-temperature views for each model stage; SFT Behavior v2 `response_text` is recorded as `continuation`.

## 100M / 2B pretrained

### Base Prompt v2 — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: France

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 is multiplied by 8.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: ices. The rain was still wet, and the sun was shining. The rain was still wet, and the sun was shining. …

### Base Prompt v2 — higher temperature

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 with 2 exponent 9.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: ices and boiling rainstorms. Once the rain began, more rain came in, and trees started to woad. …

### Behavior v2 — greedy

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: Answer:\nAnswer:\n\nThe capital of France is France.\n\nThe capital of France is France. …

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: Answer:\nAnswer:\n\nAnswer:\n\nAnswer:\n\nAnswer: …

### Behavior v2 — higher temperature

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: Answer this question\n\nWho is the capital of France?\n\nprimary boiler said that France is first, Second, Second, Third, Third …

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: The number 13+ is used to denote the number of neutrons or protons present in the periodic table. …

## 100M / 2B SFT

### Base Prompt v2 — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 multiplied by 8 is 7 multiplied by 8.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: ices. The streets were wet and cold, and the streets were wet and cold. …

### Base Prompt v2 — higher temperature

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 with 32 exponent 9.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: half- Zhong [ jurisdiction named at the 1851 hole. This grounds near Quiet Hills was in danger. …

### Behavior v2 — greedy

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: The capital of France is Paris.

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: To find the number of multiplied by 8, we need to divide 8 by 8. 8 divided by 8 equals 8. So, the answer is 8.

### Behavior v2 — higher temperature

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: The capital of France is Paris, UK.

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: To find the number of times 7 multiplied by 8 is divided by 8 … you can approximate the result by multiplying the original number by 7 …

## 100M / 10B pretrained

### Base Prompt v2 — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris\n\nQuestion: What is the name of the book that explores the history of the French Revolution? …

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 is multiplied by 8.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: iced water. The people of the village were not able to get out of the water, and the water was too cold to drink. …

### Base Prompt v2 — higher temperature

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 with 8 is 9.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: icky dirt and grey that had peered into the dense forests. This grounds the soldier sleeping in trees in front of the castle …

### Behavior v2 — greedy

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: The capital of France is France.\n\nWhat is the capital of the United Kingdom?\n\nThe capital of the United Kingdom is England. …

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: 7.5\n\nWhat is 7 multiplied by 8?\n\nAssume that the number of units of the unit …

### Behavior v2 — higher temperature

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: Average :\n\nAverage :\n\nChiasma:\n\nCluster of honor :\n\nAverage : …

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: 7.65 × 10 = 3×\nMain Question: Is 8 (1X10 = 10 or 10 X6) is abbreviated as 7 …

## 100M / 10B SFT

### Base Prompt v2 — greedy

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris\n\nQuestion: What is the main reason why the city is famous for its architecture? …

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 * 8 = 81.

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: ices. The city's streets were covered in snow, and the streets were covered in ice. …

### Base Prompt v2 — higher temperature

prompt: Question: What is the capital of France?\nAnswer:
continuation: Paris

prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation: 7 with 32 exponent\n\nQuestion: What does the text suggest for fluid movement in pipes? …

prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: ery flavonoids that blocked the sunlight, allowing heat to permeate grounds and lawns through trees in the summer …

### Behavior v2 — greedy

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: The capital of France is Paris.

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: To find 7 multiplied by 8, we need to divide the number by 8. 7 divided by 8 is 7. So, 7 multiplied by 8 is 7.

### Behavior v2 — higher temperature

prompt: Answer this question.\n\nWhat is the capital of France?
continuation: The capital of France is Paris

prompt: Answer this question.\n\nWhat is 7 multiplied by 8?
continuation: 7, 8, ..., 12, 13, 14,..., 12, 16, 18, or 17
