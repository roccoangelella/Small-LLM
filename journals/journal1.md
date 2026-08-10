Just started, first problem! Datasets are huge; terabytes of stuff. I guess we can just "stream" shards of data from huggingface instead of downloading the whole dataset. But we surely need to tokenize everything before the training happens. Anyway the vocabulary should't be that huge, at the end of the day it's just a 60-70k entries dictionary? Maybe? 

Solution came easy: we'll just use a smaller dataset. Using a big one would mean too much complexity to deal (and no compute to do that either). We'll use a "small" dataset (400gb for the pre-training, roughly 80-100b tokens), process it as a whole and cross our fingers. Instead of training on many unique tokens, we'll train for more epochs on the same tokens. The number of epochs is computed by ensuring that $E = T / D$, where $E = \text{num epochs}$, $T = \text{goal tokens}$, $D = \text{dataset tokens}$.
So, since my initial goal was to train our 1B model on 2T tokens, we'll train for 20 epochs on our 80-100 tokens. Anyway, the loss will show us the way.

Next goal is to get the split among these three datasets. 
- FineWeb-Edu has educational text, 1.3T Tokens. The "educationness" of the text is granted by HF's classifiers that kept only coherent text. It is well curated and proven to make models perform well. However the language is (by its nature of edu text) not "colloquial" enough, probably. Our model might sound like a book.
- DCLM-baseline is a 4T curated dataset that however contains potential noise
- Dolma 3 is a 6T from diverse web content. The first thing i notice is that it's full of weird porn stories and other prostitutes phone numbers, maybe we can get rid of that or pick the more colloquial stuff, if possible. Luckily text is labeled, therefore we can avoid flooding our model with porn. 
Bonus option:
- Nemotron-ClimbMix: by nvidia, it provides 400B tokens on which they trained a 1B model (dumb luck!) while outperforming equivalent models trained on mixtures of precedent datasets.

Final option (for the moment): Nemotron-ClimbMix. It removes a nice burden and provides an all-ready dataset. That'll do for now. We'll pick roughly 25% of the dataset (applying a category stratification) and cut out the whole programming text for the moment. Our #1 goal is having a model that understands english and knows stuff about general knowledge. About removing code from this dataset, gpt said this:

1. Apply an automated code-density filter to the complete selected subset.
2. Sample documents from every cluster_id and multiple shards.
3. Oversample borderline documents flagged by the filter.
4. Have an LLM classify samples as explicit code, mixed technical prose, or normal prose.
5. Manually inspect a smaller set, especially disagreements.
6. Repeat the audit after filtering and estimate the remaining code rate.
