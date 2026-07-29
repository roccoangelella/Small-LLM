Day 3! Yesterday evening i launched the production download of the dataset, and estimates told it'd take roughly 10 days. Not feasible. Bur today, as i was having breakfast, i realized we could have done it much efficiently: since we got rid of the "craft a new tokenizer" idea, we could simply stream the dataset as training is happening and, eventually (still have to decide this*) download it down to a bin file so that re-downloading it it's not necessary during the next epoch.

GPT agrees with me, so that's what we'll do. There is also an interesting problem: Nemotron's dataset is ordered by cluster id, meaning that we have a huge amount of tokens coming from cluster 1, then from cluster 2, 3,...
That is bad! If we train our model on same-domain data for that much, we're basically running a cascade of fine tunings, with our gradients running towards much diverse directions every time the domain changes, moreover, considering lr smoothing, we'd be training our model with a large lr on "cluster 1" data and with a veeeery tiny lr on "custer 20" data: the generalization abilities would be screwed! The solution to this problem comes by imposing that the dataset batches that we download are both shuffled, and, most important, representative!  We don't want only our dataset to be representative of the dataset, but we also want the batches to be so (actually, if we impose a fixed distribution on the batches, that'll be reflected on the whole dataset). 
To do that, we have to attribute a token budget to every cluster within every batch, and keep a running total of each cluster's token quota. When quota exceeded, we cut the document and place the next part into the next training batch **. That's a good a idea for a separate project as well.

**maybe that's not that efficient, we could for example enforce an overlapping budget, but that could mean overrepresenting some tokens and i'm afraid of possible consequences. Moreover, i'd like to keep the optimization level quite low now, willing at having a working thing asap, then proceed to optimize single parts.

---
In the meanwhile that our beloved agent fixes our dataset download pipeline, we can start thinking about the model's architecture. Today i'll probably go through the "preparation" of the decision process, ensuring to have a clear path of decisions to take, so that tomorrow's work is just a matter of study and decision.

I found out a new interesting concept, **Tied Embeddings**: with tied embeddings, each entry of the final hidden vector (not the output vector, as i mistakingly initially thought) is multiplied by the embedding matrix to produce logits. This product is a measure of similarity between the entry vector and the embedding vector, producing a logit that is a proxy of the "correctness" of the token in $i$-th position across the 50257 possible ones (50257 is the GPT2 vocabulary's size). With tied embeddings, we reuse that same matrix as the final vocabulary projection instead of learning a second independent matrix.

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
Let's briefly investigate how MOEs work.

An ordinary decoder block is:

$$x \to \text{Self-Attention} \to \text{FFN} \to y$$

A typical MoE block is:

$$x \to \text{Self-Attention} \to \text{Router} \to \text{Selected FFN experts} \to y$$

This means that we first process the whole input via self attention, pass the info to the router, and the router will choose among the experts to which passing the info to produce the next token. That repeats over and over for every token. 
Straightforward definition:


Experts are separate MLPs that process each token’s hidden representation when the router selects them. Their weighted outputs are added back into the residual stream and passed to the next Transformer block.


The router is usually a small linear layer that provides the score of every expert (simple classification task). The interesting part is that we don't use softmax for selection, but as weight. Experts selection uses **top k routing**: pick top k experts using router's score ranking, and the output of every expert is conveyed to a unique one by a weighted average of the output using the re-normalized output of the router (re-normalized because, unless we pick every expert, they won't sum to 1).


Amazing! So experts are just an MLP, that's why we can "Load only experts in RAM" as antirez and local llms friends say: they're just a separate network that can be safely detatched from the rest of the architecture!
### 5. Pure State Space Models
This architecture uses a State Space Model, compressing the past into a recurrent state just like RNNs do: $s_t = F(s_{t-1}, x_t)$ and $y_t = g(s_t, x_t)$. No KV cache is stored, and overall very fast inference is obtained. However, it's still quote an underground technique that for now we'll not take into account. Mamba models are the SOTA architectures.

### 6. Hybrid linear attention plus full attention
In standard attention, each new token query compares itself with every past token's key and is multiplied by each past token's value. That's quadratic and slow as hell: every token interacts with every other token, so that grows $O(L^2 d)$. The $n \times n$ matrix (where $n$ = number of tokens under analysis) is the expensive object, resulting from $Q K^\top$.

Instead, **Linear Attention** does the opposite: we first multiply $K^\top V$, which is called **memory**, denoted $S$, and then multiply the Query by the Memory to get the attention score.

Linear attention keeps the basic goal of attention: each token gathers information from other tokens, but changes how the attention weights are computed so that we never build the large $n \times n$ attention matrix resulting from $Q K^\top$ product.

With no softmax we could easily write:
$$\sum_{j} (q_i^\top k_j) v_j = q_i^\top \left( \sum_{j} k_j v_j^\top \right)$$
But softmax isn't linear and can't be fitted into both the two formulas. Moreover, running softmax on $K^\top V$ would normalize across embedding dimension and not across number of tokens. Therefore, Linear Attention needs a new way to represent similarity between tokens queries and keys without the need of using softmax function: some map functions (there is a lot of possible ones) do this for us, just allowing us to represent the attention weights by moving the summation symbol forward to the KV part, allowing us first to compute the total KV matrix, and then multiply it by $Q$, and

— i forgot that attention uses outer product, not dot product, therefore the product between each token's $k_t v_t^\top$ is a matrix, and when we sum those matrices we get the final $K^\top V$ matrix.

Done that, we multiply the $Q$ matrix (every token's query) by the "total" matrix, obtaining the attention score of each token with respect to every other in one shot, with a single matrix multiplication.
 
The convenience is straightforward: instead of having a huge $n \times n$ matrix, we first make a $d \times d$ matrix ($d$ = QKV matrices embedding vector shapes) out of $S = K^\top V$, then an $L \times d$ matrix by $Q S$.

A further efficiency step appears: we don't need to recompute KV for every token: we just keep a running total of the KV matrix that gets updated every time a new token is processed. The name "memory" makes even more sense now. However, this initial memory updating process has no obsolete information deletion process nor useful memories protection, or deciding how long information should survive: conflicting information about the same phenomenon may live together in the same space and we'd have no way to delete che old one. Stacking more and more information makes retrieval quality increasingly worse.
#### 6.1 DeltaNets
To overcome this problem, DeltaNets were introduced: we compute the "memory value" (I named it in this way), denoted $\hat{v}_t$, equal to $\hat{v}_t = S_{t-1}^\top k_t$. This determines **what the current memory associates with the new token's key**. Then, we compute the **error** as $e_t = v_t - \hat{v}_t$. That error is framed as the "useful information" that is not yet contained in our memory, therefore we update the memory by $\beta_t k_t e_t^\top$, treating $e_t$ like a "useful value". Here $\beta_t$ is a sort of learning rate, tuning the update's size.

#### 6.2 Gated DeltaNets
We supercharge deltanets with memory erasure as well, by updating the memory as:
$$S_t = \alpha_t S_{t-1} + \beta_t k_t (v_t - \alpha_t \hat{v}_t)^\top$$
As we can see we introduce the forgetting (or decay rate) denoted $\alpha_t$, that scales the entire previous memory. We also "forget" $\hat{v}_t$: equivalent to simply replacing $S_{t-1} k_t$ with $\alpha_t S_{t-1} k_t$ in the $\hat{v}_t$ formula.

#### 6.3 Kimi Delta Attention
It replaces Gated DeltaNet's $\alpha_t$ scalar with a diagonalized $\alpha_t$ vector. In this way we obtain much more powerful forgetting ability: we can control the row-wise forgetting instead of it being matrix-wise. It makes sense because each row comes from the aggregation of tokens' key and value interactions, therefore controlling our model's "memory" about different learned key features.

Importantly, the final writing of KDA is:

$$S_t = \widetilde{S}_{t-1} - \beta_t k_t k_t^\top \widetilde{S}_{t-1} + \beta_t k_t v_t^\top$$

where:

$$\widetilde{S}_{t-1} = \operatorname{Diag}(\alpha_t) S_{t-1}$$

which is precisely equivalent to:

$$S_t = \widetilde{S}_{t-1} + \beta_t k_t (v_t - \hat{v}_t)^\top$$

which is the way we always wrote it down. Much easier to digest.


Here the $\beta_t$ terms have two roles: it **erases the old association** (the product of $\beta_t$ and $\hat{v}_t$ does this, determining how much of the old key-value association should be removed) and **writes new association** (multiplying $\beta_t$ by the new value $v_t$, deciding how much to keep). To frame it visually:
- $+\beta_t k_t v_t^\top$ adds the new association between $k_t$ and $v_t$;
- $-\beta_t k_t \hat{v}_t^\top$ removes part of that old key–value association from memory.

This distinction will come in hand in the next part.
#### 6.4 Gated DeltaNet-2

Thsi paper (**may 2026! SOTA Stuff!**) splits the task using two different scalars, b for erasing and w for writing. That's smart: KDA can't choose to erase strongly but write weakly, if one is strong, so the other'll do.
 