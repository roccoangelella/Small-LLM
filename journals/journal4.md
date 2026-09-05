We're so close to start development! After studying many attention models and transformer block variants, our next step is to study Positional encoding.
The SOTA thing nearly everybody does (even though each model varies the use of that) is RoPE: **Rotary Position Embeddings**.
This technique rotates query and key vectors according to their position in the tokens sequence. 

At position $m$, RoPE applies a rotation matrix:

$$
R(m\theta) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
$$

The rotated query is $q_m^{\text{RoPE}} = R(m\theta) q$. The key at position $n$ is rotated similarly: $k_n^{\text{RoPE}} = R(n\theta) k$.

I start feeling sick of studying maths so I'll just frame RoPE as follows, and since nearly every SOTA LLM uses RoPE we'll just assume that's the best thing to do. I'll summarize it as: each token’s $Q$ and $K$ are rotated by an angle determined only by that token’s position. When two tokens are compared, the difference between their rotations encodes their relative distance.

Worth noticing that ROPE only affects attention layers that don't need memory (therefore Delta Nets layers don't use Rope, actually they don't use positional encoding at all), because these layers' recurrency already frames the token's position relevance in the Model's understanding.
---

Finally, **RMSNorm**: a normalization layer that replaces the classic LayerNorm. Instead of dividing by st. deviation + epsilon we simply scale the vector by the root mean square of its elements:

$$
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

It yields better stability, suits better with PreNorm, and is widely the standard.

### Activation Functions
Relu is old stuff. So does Gelu, appearently. Modern Decorders use **Gated FFN**. Before diving in, it's worth describing the **SiLU function**: it multiplies the input by the sigmoid of the input itself. $\operatorname{SiLU}(x) = x \cdot \sigma(x)$. The cool thing about this function is that it provides three clear regions of the activation: roughly zero, roughly 1/2 of the input signal, or roughly the signal itself, allowing a much deeper representivity than simple Relu (0 or identity function).

Now, the "real architectures":
1. GLU: Gated Linear Units: each neuron of the FFN has two weight vectors, not one. The first one, denoted $W_{\text{gate}}$, and the second, $W_{\text{up}}$. We project the input using both, and get $g$ and $u$, both being two vectors shaped equally to the current FFN layer. The output of GLU is then given by $h = \sigma(g) \odot u$. By imposing the sigmoid on $g$, we have the network deciding the amount of info should be passed, and we multiply it by $u$ to make that info pass. Basically, we make a sigmoid use the gate information to decide how much to learn and how much to forget. Then we treat $u$ as the "real info".

2. SwiGLU: it simply replaces GLU's Sigmoid with SiLU activation function. Unlike Sigmoid, SiLU isn't $[0, 1]$: it allows sign change, signal amplification, feature suppression. All the features for which we shown SiLU being better than ReLU.

Every hidden layer of every FFN will use SWiGLU. Plain and simple.
