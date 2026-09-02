---
status: evidence
observed_at: 2026-09-02
suite: eval_core_v1-full-plus-greedy32-plus-sampled
---

# 100M/10B final pretraining qualification at step 76,294

This record captures the user-supplied final qualification outputs for `100m-10b-deep-decay-from-step15500` at its planned 10B-token endpoint.

## Endpoint identity

```text
run:             100m-10b-deep-decay-from-step15500
checkpoint:      step-00076294
global step:     76,294
consumed tokens: 10,000,007,168
architecture:    gdn2_hybrid, d_model=512, d_ff=1408, 20 layers, context 2048
eval manifest:   aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792
```

The full intrinsic eval and greedy-32 behavioral run are deterministic (`temperature=0`). The supplied higher-temperature behavioral run uses `temperature=0.8`, `top_p=0.95`, `top_k=50`; this differs from the ~6.1B mid-run sampled protocol (`temperature=1.0`, `top_p=0.9`, `top_k=20`), so sampled longitudinal comparisons are descriptive rather than apples-to-apples.

## Intrinsic endpoint metrics

| metric | 100M / 2B | 100M / ~6.06B | 100M / 10B | 6.06B -> 10B | 2B -> 10B |
|---|---:|---:|---:|---:|---:|
| loss | 3.338815 | 3.167205 | 3.129107 | -0.038098 (-1.20%) | -0.209708 (-6.28%) |
| perplexity | 28.185701 | 23.741041 | 22.853570 | -0.887471 (-3.74%) | -5.332131 (-18.92%) |
| bits / decoded byte | 1.042155 | 0.988590 | 0.976699 | -0.011891 | -0.065456 |
| top-1 accuracy | 0.398875 | 0.414504 | 0.418682 | +0.004178 | +0.019807 |
| top-5 accuracy | 0.618154 | 0.638356 | 0.642991 | +0.004635 | +0.024837 |
| top-10 accuracy | 0.692041 | 0.712083 | 0.716593 | +0.004510 | +0.024552 |
| calibration ECE | 0.010091 | 0.007624 | 0.007480 | -0.000144 | -0.002611 |
| cluster macro loss | 3.349121 | 3.174749 | 3.138964 | -0.035785 | -0.210157 |
| mixture-weighted cluster loss | 3.042600 | 2.914806 | 2.871047 | -0.043759 | -0.171553 |
| worst cluster loss | 3.949742 | 3.721464 | 3.701115 | -0.020349 | -0.248627 |

Final document-bootstrap 95% loss interval: **3.097022 .. 3.159838**. The 100M/2B interval was 3.303364 .. 3.374237, so the endpoint is clearly separated from the 2B baseline. The ~6.06B interval was 3.134310 .. 3.199184 and overlaps the final interval, so the last ~3.94B tokens yield a numerically better endpoint but are not cleanly separated by simple 95% interval overlap.

The marginal loss reduction per additional billion training targets falls from about **0.0423 loss/B tokens** over 2.00B -> 6.06B to about **0.00967 loss/B tokens** over 6.06B -> 10.00B, a roughly **77% reduction in marginal improvement rate**. This is strong evidence of diminishing returns along the existing trajectory.

## Behavioral reading

Strict direct-answer reading of the 12 factual/elementary QA prompts:

```text
greedy-32 final: 4 / 12
correct: Paris, Pacific Ocean, heart, photosynthesis

sampled final (T=0.8 / p=0.95 / k=50): 6 / 12
correct: Paris, Jupiter, Pacific Ocean, Portuguese, heart, photosynthesis
```

The greedy endpoint improves from 3/12 at ~6.1B to 4/12 at 10B, but elementary factual reliability remains weak: it answers Sun for Jupiter, Earth for Mars, avoids Shakespeare, misses 0 C, does not produce yen, says Spanish for Brazil, says four days for a leap year, and fails 7x8.

Free-form generation remains mixed. The rain continuation still starts with the locally plausible `iced water`, preserving the qualitative gain over the 100M/2B endpoint, but both greedy and sampled outputs remain repetitive. The final sampled story has coherent local sentences but loops around rain/children and becomes semantically redundant. Science and encyclopedia continuations still show severe schema drift and factual corruption. Dialogue is more locally natural under sampling but drifts off topic. Structured list completion remains poor. Thus the final endpoint is a stronger language model intrinsically than 2B, while factual QA, structured induction, and long-horizon discourse remain major limitations.

## Interpretation

The 10B run achieved its endpoint and extracted additional intrinsic performance beyond the ~6.1B checkpoint, but most of the 2B -> 10B gain had already arrived by ~6B. The final ~4B tokens reduce loss by only 0.0381 versus 0.1716 over the preceding ~4.06B-token interval. The evidence therefore supports a clear **diminishing-returns/near-plateau** reading for this schedule, not continuing constant-rate improvement.

The final model is materially stronger than 100M/2B on intrinsic language modeling metrics and retains the qualitative prose-coherence improvement observed mid-run. However, 10B-token continuation alone does not solve elementary factual recall or structured prompt behavior at 100M scale.

## Preserved greedy prompt/continuation samples

### story_opening
prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: iced water. The people of the village were not able to get out of the water, and the water was too cold to drink.\n\nThe village was

### science_explanation
prompt: Water can exist as a solid, a liquid, or a gas. The transition from liquid water to water vapor occurs when 
continuation: __________.\n\nQuestion: What is the difference between a liquid and a gas? Answer: A liquid is a solid, while a gas is a liquid

### encyclopedia_style
prompt: The Roman Republic was a period of ancient Roman civilization that began after 
continuation: vernacular Roman times. The Roman Empire was a major part of the Roman Empire, and was the largest and most powerful empire in the world. The Roman Empire

### dialogue
prompt: Alice: Did you remember to close the window?\nBen: I thought you had closed it.\nAlice:
continuation:  I didn't.\nBen: I didn't.\nAlice: I didn't.\nBen: I didn't.\nAlice: I didn't.

### list_pattern
prompt: France | Paris\nItaly | Rome\nGermany |
continuation:  Germany\nFrance | France\nFrance | France\nFrance | France\nFrance | France\nFrance | France\nFrance | France\nFrance | France\nFrance |

### sentiment_pattern
prompt: Text: I loved every minute of the film.\nSentiment: positive\n\nText: The plot was tedious and predictable.\nSentiment: negative\n\nText: The acting was excellent, although the ending was weak.\nSentiment:
continuation:  bad\n\nText: The plot was boring and the ending was not good.\nSentiment: bad\n\nText: The plot was boring and the ending

### capital_france
prompt: Question: What is the capital of France?\nAnswer:
continuation:  Paris\n\nQuestion: What is the name of the book that explores the history of the French Revolution?\nAnswer: The French Revolution: The Revolution of the

### largest_planet
prompt: Question: What is the largest planet in the Solar System?\nAnswer:
continuation:  The Sun

### red_planet
prompt: Question: Which planet is commonly called the Red Planet?\nAnswer:
continuation:  The Earth

### hamlet_author
prompt: Question: Who wrote the play Hamlet?\nAnswer:
continuation:  The play Hamlet was written by the author of the play Hamlet.\n\nQuestion: What is the main topic of the text? Answer: The text

### water_freezing
prompt: Question: At what temperature does pure water freeze on the Celsius scale?\nAnswer:
continuation:  The Celsius scale is a temperature range of -40°C to 100°C.\n\nQuestion: What is the temperature at which the water in a glass

### largest_ocean
prompt: Question: What is the largest ocean on Earth?\nAnswer:
continuation:  The Pacific Ocean

### japan_currency
prompt: Question: What is the currency of Japan?\nAnswer:
continuation:  The currency of Japan is the Japanese currency.

### brazil_language
prompt: Question: What is the main language spoken in Brazil?\nAnswer:
continuation:  Spanish

### blood_pump
prompt: Question: Which organ pumps blood through the human body?\nAnswer:
continuation:  The heart\n\nQuestion: What is the primary function of the heart?\nAnswer: The heart pumps blood through the human body.

### photosynthesis
prompt: Question: What process do plants use to convert light energy into chemical energy?\nAnswer:
continuation:  Photosynthesis\n\nQuestion: What is the process by which plants convert light energy into chemical energy?\nAnswer: Photosynthesis

### leap_year
prompt: Question: How many days are in a leap year?\nAnswer:
continuation:  Four days

### simple_arithmetic
prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation:  7 is multiplied by 8.

## Preserved higher-temperature prompt/continuation samples

### story_opening
prompt: The rain had stopped before dawn, leaving the streets covered in 
continuation: iced water.\n\nThe children were delighted to see the rain and saw the sun shining through their window, and the rainwater melted away.\n\nThe children had to watch carefully as the rain began to build.\n\nIt was raining but the rain was still…

### science_explanation
prompt: Water can exist as a solid, a liquid, or a gas. The transition from liquid water to water vapor occurs when 
continuation: ____ water is in the mixture, which can be written as ____ or as ____ (the same as ____).\n\n1. Which one of the following is NOT a characteristic of water?\n\n(a) It has a boiling point of 40,000K\n\n(b) It has a freezing point of 10,000K\n\n…

### encyclopedia_style
prompt: The Roman Republic was a period of ancient Roman civilization that began after 
continuation: __________.\nWhat is the most important historical period for children?\nWhat is the most important historical period for children?\nWhat is the most important historical period in the history of America?\nWhat is the most important historical…

### dialogue
prompt: Alice: Did you remember to close the window?\nBen: I thought you had closed it.\nAlice:
continuation:  Oh, I think so.\nBen: You have to have a camera to take pictures.\nAlice: And now the kids were all up in the air with their cameras and they were working on something.\nAlice: I mean, I think that's the first thing that we are going to do with…

### list_pattern
prompt: France | Paris\nItaly | Rome\nGermany |
continuation:  Sweden | Sweden\nItaly | Italy\nItaly | Spain | Spain\nItaly | Germany | Germany\nItaly | Italy\nItaly | Sweden | Sweden\nItaly | Italy\nItaly | Hungary\nItaly | Austria\nItaly | Hungary\nItaly | Hungary\nItaly | Germany | Germany\nItaly | Germany\nItaly | Spain\n

### sentiment_pattern
prompt: Text: I loved every minute of the film.\nSentiment: positive\n\nText: The plot was tedious and predictable.\nSentiment: negative\n\nText: The acting was excellent, although the ending was weak.\nSentiment:
continuation:  positive\n\nText: I liked the story and the overall plot.\nSentiment: negative\n\nText: The story was easy to follow and focused on the plot.\nSentiment: positive\n\nText: The plot was easy

### capital_france
prompt: Question: What is the capital of France?\nAnswer:
continuation:  Paris\n\nQuestion: Which of the following is NOT a type of fishing caught on a fishing boat, according to the text?\nAnswer: Fishing on a boat\n\nQuestion: What is the main purpose of a fishing boat, according

### largest_planet
prompt: Question: What is the largest planet in the Solar System?\nAnswer:
continuation:  Jupiter

### red_planet
prompt: Question: Which planet is commonly called the Red Planet?\nAnswer:
continuation:  Earth

### hamlet_author
prompt: Question: Who wrote the play Hamlet?\nAnswer:
continuation:  William the Conqueror\n\nQuestion: What is the title of the play that was originally published? Answer: Hamlet

### water_freezing
prompt: Question: At what temperature does pure water freeze on the Celsius scale?\nAnswer:
continuation:  Under 30°C (77°F)\n\nQuestion: What is the melting point of water?\nAnswer: 1,859°C (2,859°F)\n\nQuestion: Which of the following is NOT

### largest_ocean
prompt: Question: What is the largest ocean on Earth?\nAnswer:
continuation:  The Pacific Ocean

### japan_currency
prompt: Question: What is the currency of Japan?\nAnswer:
continuation:  The currency of Japan is Japanese.\n\nQuestion: What is the primary source of energy for the engine in a car? Answer: Gasoline or diesel fuel.

### brazil_language
prompt: Question: What is the main language spoken in Brazil?\nAnswer:
continuation:  Portuguese

### blood_pump
prompt: Question: Which organ pumps blood through the human body?\nAnswer:
continuation:  The heart\n\nQuestion: What is the primary function of the heart?\nAnswer: The heart pumps blood through the human body.

### photosynthesis
prompt: Question: What process do plants use to convert light energy into chemical energy?\nAnswer:
continuation:  Photosynthesis, where the plant cells use light to convert glucose into oxygen and oxygen into carbon dioxide.

### leap_year
prompt: Question: How many days are in a leap year?\nAnswer:
continuation:  5 days

### simple_arithmetic
prompt: Question: What is 7 multiplied by 8?\nAnswer:
continuation:  7 is multiplied by 8, which is 8 times 7 times the square root of 6.

## Sources

- User-supplied `100m-10b-step76294-eval-core-full-greedy.json`.
- User-supplied `100m-10b-step76294-greedy32.json`.
- User-supplied `100m-10b-step76294-sampled.json`.
- Prior project evidence: `100m_10b_midrun_6b_evaluation_2026-08-24.md`.
- Prior project evidence: `20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`.
