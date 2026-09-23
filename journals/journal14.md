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

***

## Our Chosen Architecture: The Small-LLM MoE Recipe

After studying the literature, running the numbers, and debating the trade-offs with Edo, we converged on the exact MoE recipe we're going to implement for our ultra-small model.

Our guiding philosophy here was simple: **isolate the MoE dynamics**. We don't want to change five things at once and then wonder which one made the run fail. We already have a solid, well-behaved reference: our 20M dense Small-LLM backbone (8 decoder blocks, $d_{\mathrm{model}} = 256$, SwiGLU, dense $d_{\mathrm{ff}} = 704$). So we decided to use that exact backbone as our baseline.

Here is the blueprint of what we're building and why:

### 1. Placement and Geometry: 64 Experts, Top-2
- **Full MoE replacement**: All 8 decoder blocks replace their dense SwiGLU FFN with an MoE block. Attention mixers, RMSNorm, and residual streams stay untouched.
- **64 routed experts ($E = 64$), Top-2 routing ($K = 2$)**.
- **No shared experts ($0$ shared)** for now: keep it simple and clean.
- **Dropless routing**: No token dropping, no capacity limits, no tokens sent to the void. Every token gets its full Top-2 experts processed.
- **The $K \cdot h$ invariance trick**: Each expert is an ordinary bias-free SwiGLU with hidden width $h = 352$.
  Notice why $h = 352$:

```math
K \cdot h = 2 \cdot 352 = 704
```

  $704$ is the exact active FFN width of our 20M dense model! This is super neat: every single token requires the exact same active FLOPs and compute through the FFN as the dense model. Yet, our total parametric capacity per layer jumps to $E \cdot h = 64 \cdot 352 = 22,528$! We get the massive parameter capacity of a much larger model at the inference and training cost of our tiny 20M dense model.
  (And down the road, we can test finer granularities like 128 experts / Top-4 / $h=176$ or 256 experts / Top-8 / $h=88$ while keeping both active compute $K \cdot h = 704$ and total capacity $E \cdot h = 22,528$ locked).

### 2. Router Scoring: Sqrt(Softplus)
Instead of standard Softmax or Sigmoid, we adopted DeepSeek-V4's approach:

```math
s_i = \sqrt{\mathrm{softplus}(z_i)} = \sqrt{\ln(1 + e^{z_i})}
```

where $z = W_r x$ are the router logits.
Why?
Sigmoid caps affinities strictly between $0$ and $1$. When a token has a super strong affinity for an expert, sigmoid saturates and pushes gradients towards zero, slowing down specialization. Softplus provides a smooth, non-negative activation that doesn't artificially hit a ceiling at 1. The square root then prevents the activations from blowing up linearly for large logits, keeping gradient flow well-behaved without freezing the router.

### 3. Selection vs. Mixture Weights
This is a subtle but crucial distinction that trips up a lot of people when looking at loss-free balancing:
1. **Expert Selection**: We pick the Top-2 experts using the bias-adjusted scores:

```math
\mathrm{Selected} = \mathrm{Top2}(s + b)
```

2. **Mixture Weighting**: Once the Top-2 experts are picked, we **throw away the bias $b$**! The actual combination weights are computed by normalizing the raw, unbiased positive scores $s$ over the chosen experts:

```math
w_i = \frac{s_i}{\sum_{j \in \mathrm{Selected}} s_j}
```

Why do this? Because $b$ is purely a traffic-management tool. It exists to force the router to explore underused experts and keep the workload balanced across hardware. But once an expert is selected, we want its contribution to the actual token representation to reflect its true semantic score $s$, not our artificial traffic bias. Tainting the output logits with $b$ would distort the representation and hurt language modeling performance.

### 4. Step-Level Quantile Balancing (and why not per-microbatch)
We implement Kimi K3's Quantile Balancing (QB), but we made a deliberate choice on when to update the bias $b$:
- In training, we use gradient accumulation (multiple microbatches make up one logical optimizer step).
- We keep the bias $b_t$ **frozen across all microbatches** belonging to optimizer step $t$.
- We accumulate routing statistics across all those microbatches, and only at the very end of the step do we compute the exact quantile cutoff, update to $b_{t+1}$, and mean-center it.

Why not update $b$ after every microbatch? If you update $b$ inside the accumulation loop, microbatch 1 and microbatch 4 in the same step would be routing under different policies, introducing high-frequency noise into the accumulated gradients. Furthermore, pooling tokens across the entire optimizer step gives us a much larger, less noisy sample to accurately estimate the quantile threshold $\alpha$.

### 5. What's Z-Loss, and Why Don't We Need It?
In many classical MoE papers (like ST-MoE or PaLM), you'll see something called **router z-loss**:

```math
L_z = c_z \cdot \frac{1}{T} \sum_{t=1}^T \left( \ln \sum_{i=1}^N e^{z_{t,i}} \right)^2
```

What does it actually do?
In softmax routers, the router computes $\frac{e^{z_i}}{\sum e^{z_j}}$. During training, the router can drift into outputting enormous logits ($z \gg 0$). While adding a constant to all logits doesn't change softmax probabilities mathematically, in half-precision floating point (like FP16 or BF16), $e^z$ will overflow to `inf` or `NaN` very quickly. Moreover, huge logits make the router hyper-confident and stiff. The z-loss penalizes large log-partition values $(\ln \sum e^z)^2$, forcing router logits to stay reasonably close to zero and numerically stable.

So why aren't we using z-loss in our model?
1. **No Softmax Partition Function**: We use $\sqrt{\mathrm{softplus}(z)}$. We never compute $\sum e^z$, so there is no exponential sum to blow up.
2. **Pure FP32 Router Path**: We keep all router operations (logits $z$, scores $s$, margins, top-k selection, and mixture weights) strictly in FP32. No FP16 underflow/overflow risks.
3. **Softplus + Sqrt is naturally well-behaved**: The combination naturally dampens extreme values without needing an extra loss term.

Instead of adding another hyperparameter and another auxiliary gradient to fight the main cross-entropy loss, we simply track router health directly via W&B telemetry (logit RMS, max absolute values, gradient norms, and QB bias drift). If the metrics stay clean, we don't need the crutch.

### 6. Initialization and Optimizer
- **Router weights**: We initialize $W_r \sim \mathcal{N}(0, 0.02^2)$, matching the standard GPT-style normal initializer of our dense backbone. With $d_{\mathrm{model}} = 256$, the initial logits have a standard deviation of $\approx \sqrt{256} \cdot 0.02 = 0.32$. This lands our initial logits right in the responsive, near-linear sweet spot of $\sqrt{\mathrm{softplus}}$, breaking symmetry gently without saturating.
- **Bias**: The non-gradient QB bias $b$ starts at exactly $0$.
- **Optimizer**: AdamW for the router with weight decay set to $0$ and a $1.0\times$ LR multiplier, keeping it in sync with our baseline hybrid Muon+AdamW optimizer setup.

With this architecture, we get the best of both worlds: the clean, interpretable foundation of our 20M dense model, paired with modern, loss-free Top-2 routing that gives us massive parameter capacity while keeping the FLOP budget practically unchanged. Next step: get to coding and qualify the routing telemetry!