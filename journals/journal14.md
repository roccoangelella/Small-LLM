After the 100M/10B run ended I started working on this project collaboratively with my friend Edo. He made me notice that, provided the amount of tokens we're starting to work with, there was maybe no point in rejecting the MOE architecture, which could provide extre free performance with almost no extra compute needed.

Therefore it's time to delve into what MOEs are, how they work, which techniques are used by sota models.

***

## What's a MOE?
When I started studying this topic I thought it was much more complex, while at the end of the day it's "just" using many "specialized" MLPs instead of a single one, with a router running a classification task on which MLP should be the one processing the output of our attention blocks. The router ranks the most likely experts, and we inject in the top-$k$ experts the attention's output, using only those in the next token preiciton part. Usually by taking the weighted average of the logits, weighting each expert's output by the probability that the router's softmax assigned to them.

That's basically it, in principle, but in practice, there's a lot of stuff we have to do to make sure our experts learn correctly, and most important, that our router doesn't produce a largely skewed distribution, preferring to send data to a leading expert rather than distributing them more evenly.

It's worth remembering that experts won't be "domain experts", in the sense that E1 won't be the one who knows maths, E2 code,... These are black box-attention generated features. However, that still makes a great mental model.

### Loss-Free balancing:
This is the main technique used to preved heavy skeweness in Router's distribution: we introduce a non-learned Bias, denoted $b_i$, that we add to the router's softmax output. The bias is constantly updated so that heavily routed experts receive a smaller bias and will therefore need a very strong router signal to be designated as the best expert.

### Differentiation 
The output of a plain top-$k$ operation isn't differentiable, being a discrete operation. To introduce differentiability we compute the MOE's output as a weighted average of the Experts' outputs weighted by the router's softmax probabilities, as we said above. This allows us to differentiate and make the gradients flow.

### Sigmoid routing:
Softmax enforces the experts to be in a "free for all" competition. This means that the softmax routing process isn't an independent evaluation process. To overcome this dependency bias, sota LLMs have the router producing $N$ sigmoids, ranking according to the 0-1 score. Only after picking the top-$k$, the sigmoids output is normalized to produce a probability distribution.

### Softplus routing: 
Deepseek-V4 doesn't use sigmoid routing but $\sqrt{\mathrm{softplus}(\cdot)}$. Softplus is a continuous approximation of the ReLU function. It follows that, by taking $\sqrt{\mathrm{softplus}(a)}$, we avoid being constrained in a 0-1 range (which is also the historical reason for ReLU taking over sigmoid, other than higher computational ease), and we're able to express activations that are much higher than 1 (despite in an increasing marginal way due to the sqrt).

***

## Experts Collapse
As we already said, when an expert gets MUCH better than others, the router might get heavily biased towards that expert, and give us an output distribution that looks like: 
[52%,18%,10%,7%,5%,4%,3%,1%]

This is a problem for mainly two reasons, one is more obvious and the other one is fascinating:
1. If one expert is loaded with knowledge and the other ones are basically useless, what's the point of using experts? Answer: no point in this unbalanced MOE distribution.
2. When distributing training across many devices, the device storing the "hot" expert will receive all the tokens. The other device will receive (almost) none, using expensive GPUs to store an expert that'll never get used.

### Auxiliary load-balancing loss:
For every expert $i$ we define:

```math
f_i = \frac{1}{T} \sum \mathbf{1}[\mathrm{argmax}(p(x)) = i]
```

that simply defines, among the total number of tokens, the percentage of which had ben routed to expert $i$.
Then, we also compute the average probability that the router assigns a token to expert $i$ as:

```math
P_i = \frac{1}{T} \sum p_i(x)
```

We use them to compute:

```math
L_{\mathrm{balance}} = \alpha \cdot N \sum f_i P_i
```

This is called **Balance Loss**. $\alpha$ is just a coefficient, commonly set at $10^{-2}$.


This Balance Loss will be summed to the CE loss, and the reason is straightforward: we introduce a loss that is inversely proportional to the entropy of the router distribution: higher entropy (and therefore nearly equal probabilities of being routed in either of the $i$ experts) leads to lower loss, while preferring an expert over the others lowers the entropy and triggers a higher loss, impressing this as an unwanted behavior in the router. (I speak in terms of entropy mainly as a mental model, entropy isn't mearued here).

However, using this loss is equivalent to forcing the router to learn both which expert is the best one for each task and to use experts in a uniform way. This can generate conflict and harm the performances of the router, and therefore of the whole llm.

### Auxiliary-loss-free balancing:
We introduce a non-learned bias that is inversely proportional to the expert's charge, defined as:

```math
b_i^{t+1} = b_i^t + \gamma \mathrm{sign}(\bar{c} - c_i)
```

This bias is then added to each expert $i$ during the Top-$K$ selection phase $T = \mathrm{TopK}(s + b, K)$. This technique has been proved to yield a much better experts balancing then Auxiliary loss method.

DeepSeek-V4 uses a mixture of these two techniques. It mostly relies on bias, but it also uses a tiny balance loss weight, having proved that it can help avoiding extreme imbalances.

### Quantile Balancing:
Used by Kimi K3, running the top 16 out of 896 experts. For a batch of $m$ tokens, $mK$ routing decisions will take place. If we're willing to distribute them across 896 experts, we'd need $q = \frac{mK}{N}$ tokens per expert, which, for K3, means that each expert is selected to process roughly 1.786% of the $m$ tokens.
Kimi K3 processes tokens in this way: 
1. The router computes the Sigmoid scores for every expert, producing 896 scores for every token, applying the bias to the Top-$K$ ranking.
2. Once the top 16 is selected, the Sigmoid scores get normalized (with no bias!!), willing to use them for the 16-items weighted average.
3. Interestingly, Kimi saves also the Sigmoid+bias score of the $K+1$ expert as well, denoted $\alpha$, and called **cutoff**. It describes the threshold that an expert must overcome to get into the Top-$K$ for token $i$.
4. Cutoff $\alpha$ is used to compute the *margin*, denoted:

```math
m_{i,j} = s_{i,j} - \alpha_i
```

where $i \to \text{token}; j \to \text{expert}$. It frames how far each expert is from that token's cutoff. The expert $j$ is above token $i$'s cutoff when $s_{i,j} + b_j > \alpha_i$, which is equivalent to:

```math
m_{i,j} > -b_j
```

which is the central equation here.
5. At this point, the bias is basically the margin's threshold: we pick experts whose margin is greater than bias, therefore we can count the number of tokens assigned to expert $j$ as:

```math
\ell_j(b_j) = \sum_{i=1}^m \mathbf{1}[m_{i,j} > -b_j]
```

that we want to be equal to $q$, therefore the final goal is:

```math
\sum_{i=1}^m \mathbf{1}[m_{i,j} > -b_j] = \frac{mK}{N}
```

To obtain this we run a simple optimization task that finds the optimal bias.