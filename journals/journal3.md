Day 3! Yesterday evening i launched the production download of the dataset, and estimates told it'd take roughly 10 days. Not feasible. Bur today, as i was having breakfast, i realized we could have done it much efficiently: since we got rid of the "craft a new tokenizer" idea, we could simply stream the dataset as training is happening and, eventually (still have to decide this*) download it down to a bin file so that re-downloading it it's not necessary during the next epoch.

GPT agrees with me, so that's what we'll do. There is also an interesting problem: Nemotron's dataset is ordered by cluster id, meaning that we have a huge amount of tokens coming from cluster 1, then from cluster 2, 3,...
That is bad! If we train our model on same-domain data for that much, we're basically running a cascade of fine tunings, with our gradients running towards much diverse directions every time the domain changes, moreover, considering lr smoothing, we'd be training our model with a large lr on "cluster 1" data and with a veeeery tiny lr on "custer 20" data: the generalization abilities would be screwed! The solution to this problem comes by imposing that the dataset batches that we download are both shuffled, and, most important, representative!  We don't want only our dataset to be representative of the dataset, but we also want the batches to be so (actually, if we impose a fixed distribution on the batches, that'll be reflected on the whole dataset). 
To do that, we have to attribute a token budget to every cluster within every batch, and keep a running total of each cluster's token quota. When quota exceeded, we cut the document and place the next part into the next training batch **. That's a good a idea for a separate project as well.

**maybe that's not that efficient, we could for example enforce an overlapping budget, but that could mean overrepresenting some tokens and i'm afraid of possible consequences. Moreover, i'd like to keep the optimization level quite low now, willing at having a working thing asap, then proceed to optimize single parts.

---
In the meanwhile that our beloved agent fixes our dataset download pipeline, we can start thinking about the model's architecture. Today i'll probably go through the "preparation" of the decision process, ensuring to have a clear path of decisions to take, so that tomorrow's work is just a matter of study and decision.

I found out a new interesting concept, ** Tied Embeddings **: with tied embeddings, each entry of the final hidden vector (not the output vector, as i mistakingly initially though) is multiplied by the embedding matrix to produce logits. This product is a measure of similarity between the entry vector and the embedding vector, producing a logit that is a proxy of the "correctness" of the token in i-th position across the 50257 possible ones (50257 is the GPT2 vocabulary's size). With tied embeddings, we reuse that same matrix as the final vocabulary projection instead of learning a second independent matrix.

that said, we can start with our model geometry investigation:

# **Dense vs MOE vs Whatever**
### 1. The Dense Softmax Attention Transformer

The most classic llm architecture is the "Dense softmax-attention Transformer". that's what i think of when i think about an llm. That says it all on how old my knowledge is. Old, full attention, every token sees every token. That's basically GPT-2.

Compared with GPT-2, modern models commonly use:
- RMSNorm instead of LayerNorm;
- pre-normalization;
- RoPE instead of learned absolute positions;
- SwiGLU instead of GELU MLPs;
- GQA instead of ordinary MHA;
- tied input/output embeddings at small scales;
- few or no linear biases;
- zero or very low dropout;
- sometimes QK-Norm;
- greater depth relative to width.

We'll see these improvings in next stages.

### 2. Dense Transformer with local and global attention
INstead of running full attentino between every token, we use a sliding window and un attention only in the window, running full attentino only in some layers, interspersed with a fixed period. I think many modern architectures (i'm pretty sure Modern BERT does this) use this technique. 
The main benefit is reducing KV cache ***, and getting faster and cheaper inference. Using it with a 2048 context yields very negligible benefits, but it's still an interesting technique that we must takeinto account.

### 3. Layer-shared dense Transformer
Basically a cheat that has been proved to yield a tiny improvement: we duplicate each transformer layer sequentially: layer 1->layer1->layer2->layer2->... 
Doesn't reduce params nor training time, just a bit of computation. Interesting workaround, would be interesting to check wether it's just an inference setting or has an influence on training. (asnwer: during training phase the model is doubled as well)

### 4. Mixture-of-Experts Transformer
The SOTA technique for large models. Surely unattractive for our project: using experts that small would probably mean make them very dumb.

### 5. Pure State Space Models
This architecture uses a State Space Model, compressing the past into a recurrent state just like RNNs do. s_t=F(s_t-1,x_t) and y_t=g(s_t,x_t). No KV cache is stored, and overall very fast inference is obtained. However, it's still quote an underground technique that for now we'll not take into account. Mamba models are the SOTA architectures.

### 6. Hybrid linear attention plus full attention
