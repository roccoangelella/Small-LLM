After the 100M/10B run ended I started working on this project collaboratively with my friend Edo. He made me notice that, provided the amount of tokens we're starting to work with, there was maybe no point in rejecting the MOE architecture, which could provide extre free performance with almost no extra compute needed.

Therefore it's time to delve into what MOEs are, how they work, which techniques are used by sota models.

***

## What's a MOE?
When I started studying this topic I thought it was much more complex, while at the end of the day it's "just" using many "specialized" MLPs instead of a single one, with a router running a classification task on which MLP should be the one processing the output of our attention blocks. The router ranks the most likely experts, and we inject in the top-$k$ experts the attention's output, using only those in the next token preiciton part. Usually by taking the weighted average of the logits, weighting each expert's output by the probability that the router's softmax assigned to them.

That's basically it, in principle, but in practice, there's a lot of stuff we have to do to make sure our experts learn correctly, and most important, that our router doesn't produce a largely skewed distribution, preferring to send data to a leading expert rather than distributing them more evenly.

It's worth remembering that experts won't be "domain experts", in the sense that $E_1$ won't be the one who knows maths, $E_2$ code,... These are black box-attention generated features. However, that still makes a great mental model.

### Loss-Free balancing:
This is the main technique used to preved heavy skeweness in Router's distribution: we introduce a non-learned Bias, denoted $b_i$, that we add to the router's softmax output. The bias is constantly updated so that heavily routed experts receive a smaller bias and will therefore need a very strong router signal to be designated as the best expert.

### Differentiation 
The output of a plain top-$k$ operation isn't differentiable, being a discrete operation. To introduce differentiability we compute the MOE's output as a weighted average of the Experts' outputs weighted by the router's softmax probabilities, as we said above. This allows us to differentiate and make the gradients flow.

### Sigmoid routing:
Softmax enforces the experts to be in a "free for all" competition. This means that the softmax routing process isn't an independent evaluation process. To overcome this dependency bias, sota LLMs have the router producing $N$ sigmoids, ranking according to the 0-1 score. Only after picking the top-$k$, the sigmoids output is normalized to produce a probability distribution.

### Softplus routing: 
Deepseek-V4 doesn't use sigmoid routing but $\sqrt{\operatorname{softplus}(\cdot)}$. Softplus is a continuous approximation of the ReLU function. It follows that, by taking $\sqrt{\operatorname{softplus}(a)}$, we avoid being constrained in a 0-1 range (which is also the historical reason for ReLU taking over sigmoid, other than higher computational ease), and we're able to express activations that are much higher than 1 (despite in an increasing marginal way due to the sqrt).

***

## Experts Collapse
As we already said, when an expert gets MUCH better than others, the router might get heavily biased towards that expert, and give us an output distribution that looks like: 
[52%,18%,10%,7%,5%,4%,3%,1%]

This is a problem for mainly two reasons, one is more obvious and the other one is fascinating:
1. If one expert is loaded with knowledge and the other ones are basically useless, what's the point of using experts? Answer: no point in this unbalanced MOE distribution.
2. When distributing training across many devices, the device storing the "hot" expert will receive all the tokens. The other device will receive (almost) none, using expensive GPUs to store an expert that'll never get used.

### Auxiliary load-balancing loss:
For every expert $i$ we define:
$$f_i = \frac{1}{T} \sum \mathbf{1}[\operatorname{argmax}(p(x)) = i]$$
that simply defines, among the total number of tokens, the percentage of which had ben routed to expert $i$.
Then, we also compute the average probability that the router assigns a token to expert $i$ as:
$$P_i = \frac{1}{T} \sum p_i(x)$$
We use them to compute:
$$L_{\text{balance}} = \alpha \cdot N \sum f_i P_i$$ 
This is called <u>Balance Loss</u>. Alpha il just a coefficient, commonly set at 10^-2.


This Balance Loss will be summed to the CE loss, and the reason is straightforward: we introduce a loss that is inversely proportional to the entropy of the router distribution: higher entropy (and therefore nearly equal probabilities of being routed in either of the i experts) leads to lower loss, while preferring an expert over the others lowers the entropy and triggers a higher loss, impressing this as an unwanted behavior in the router. (I speak in terms of entropy mainly as a mental model, entropy isn't mearued here).

However, using this loss is equivalent to forcing the router to learn both which expert is the best one for each task and to use experts in a uniform way. This can generate conflict and harm the performances of the router, and therefore of the whole llm.

### Auxiliary-loss-free balancing:
We introduce a non-learned bias that is inversely proportional to the expert's charge, defined as bi^t+1​=bi^t+γsign(cˉ−ci​). This bias is then added to each expert i during the Top-K selection phase T=TopK(s+b,K). This technique has been proved to yield a much better experts balancing then Auxiliary loss method.

DeepSeek-V4 uses a mixture of these two techniques. It mostly relies on bias, but it also uses a tiny balance loss weight, having proved that it can help avoiding extreme imbalances.

### Quantile Balancing:
Used by Kimi K3, running the top 16 out of 896 experts.