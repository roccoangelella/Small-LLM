Aaaaand we are live!

The first pre training process is currently (quite smoothly though!) flowing from a 10.8 to a 8.0 cross entropy loss during the first 50 updates. Validation loss is 7.91 at this stage. Nice! Perplexity is 2739.35 at this point.

What the hell is perplexity? I always asked it myself! It's just the exponential of the cross entropy loss. It's interpretation is, roughly: 
**How many tokens is my model scanning through, to choose the correct one?** Very simply, if the model has $\text{loss} = 0$, $e^0 = 1$, meaning it is considering only one token, the correct one.

When we had loss 7.91, our model was uncertain about 2739.35 tokens. Plain and simple. Considering that the model has to pick among roughly 50k tokens, that's still something.

I noticed we're using only 32% of GPU during this run. Might be useful to try microbatch=2,3,4,5... in next runs.

100 updates: perplexity=769!

---
Training ended, and it behaves surprisingly well! I mean, it speaks! It speaks nonsense, but that's still something! The next trials will simply see us enlarging the dataset and the model, iterating until a solid speaking ability comes out. We'll probably consider as "good" a perplexity level close to 20. We ended with rouhgly 420.

I feel like we should bump the dataset size rather than the model size for the moment, to isolate wether the "ability of talking" comes rather from seeing enough tokens or it's a feature that small models can't posses. In the few months of experience i have with ML, i consider to enlarge the model only when underfitting shows up. GPT agrees. We'll switch to 100M tokens now.

---
While 100M trains, we'll deal with the test suite. We already made some simple "behavioral" tests, now we'll also implement a more rigorous test. Done that, we'll probably set up the first Supervised Fine Tuning training module, to make the model chat. I wonder wether that'll work with our ancestral 100M model.

I guess that the next step won't be to keep enlarging the model (provided that this run shows some better context understanding and sentente-completion capabilities) but rather to run the next training step (probably SFT), to see how the model behaves on a smaller scale. Pre training the big model and running next steps on that (much much slower) would make our pipeline slow down by days.

Reearches suggest that we should wait at least for the **Chinchilla reference**, which is 20 tokens/parameter. We'll wait for the facts to prove that our 100M can't write properly, then we'll decide.
 
*A particularly relevant July 2026 study trained models from 5M to 1B parameters through pretraining, reasoning SFT and RL with verifiable rewards. It included several 20M-parameter runs. Better-pretrained checkpoints consistently produced better post-SFT models, higher eventual RL ceilings and faster improvements under RL. Pretraining loss was strongly predictive of later post-training performance.* (source: https://arxiv.org/html/2607.16097v1)

---
Interesting finding: during SFT, we distinguish between "simple tokens" and "loss-bearing tokens": unlike pre training, where we train our model in predicting every token in the input sequentially, the effective train happens only on the part of the input that contains the answer to the user question, which is what we're interested in. Therefore, the loss is only ocmputing using the model's prediction of that part.

SFT strategy: we'll split the SFT in three steps:

- **S0**: Conversational training: we teach our model how to answer prompts, using for the 85% huggingface's Smol-SmolTalk, and 15% will still come from ClibMix dataset, to avoid the model loosing semantic relationships. Inside this 85% we'll have :
1. smol-magpie-ultra-short: core component of the dataset, containing mainly conversations and Q&A.
2. smol-contraints: constrained Q&A, like bullet points, using placeholders like <your name> and stuff like that.
3. smollm-rewrite-30k: teaching to rephrase concepts.
4. smol-summarize-20k: teaching to summarize concepts.

