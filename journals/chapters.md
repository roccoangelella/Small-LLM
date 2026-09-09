# Journals Index

Quick navigation index across all project research and engineering journals.

| Chapter | Title / Topic | Summary |
| :--- | :--- | :--- |
| [Journal 1](journal1.md) | **Dataset Selection & Sizing** | Sizing token budgets (1B/2T vs 80–100B tokens), comparing FineWeb-Edu, DCLM, Dolma 3, Nemotron-ClimbMix, and designing code-density filters. |
| [Journal 2](journal2.md) | **Data Pipeline & Tokenization** | Detokenization vs reusing GPT-2 BPE tokens, Nemotron cluster ID reliability, and packing data into binary `.bin` token shards. |
| [Journal 3](journal3.md) | **Architecture Exploration** | Comparing Dense Transformers, MoE, SSM/Mamba, and Linear Attention. Deep dive into Tied Embeddings, DeltaNets, Gated DeltaNet-2 (GDN-2), and KDA. |
| [Journal 4](journal4.md) | **Positional Encoding & Layer Components** | Rotary Position Embeddings (RoPE), RMSNorm vs LayerNorm, SiLU activation, GLU, and SwiGLU FFN blocks. |
| [Journal 5](journal5.md) | **Optimizers & Training Dynamics** | Muon (Newton-Schulz orthogonalization, SVD intuition, head-wise scaling) split with AdamW, and SOAP eigenbasis optimizer. |
| [Journal 6](journal6.md) | **Pre-Training Qualification** | 20.6M GDN-2 hybrid geometry (3 GDN-2 + 1 full attention), 10M qualification dataset build, manifest verification, and 306-update plan. |
| [Journal 7](journal7.md) | **First Run Live & SFT Roadmap** | First 10M run telemetry, understanding perplexity, scaling to 100M tokens, Chinchilla reference ratio, and S0 conversational SFT staging. |
| [Journal 8](journal8.md) | **Speed Bottlenecks & Chunking** | Token throughput collapse (3,000 to 400 tok/s), GDN-2 learned decay, exploding exponentials ($e^{G_j - G_i}$), and chunk-shrinking limits. |
| [Journal 9](journal9.md) | **FLA Kernel Optimization** | Switching to Fast Linear Attention (FLA) reaching 20k tok/s, launching 2B run on 20M model, CPU-GPU sync bottlenecks vs fused Triton kernels. |
| [Journal 10](journal10.md) | **20M/2B vs 100M/2B & Sampling** | Emergence of factual knowledge, greedy repetition loops vs stochastic sampling (temp=1, top-p=0.9, top-k=20), and Kaggle dual-T4 setup. |
| [Journal 11](journal11.md) | **Reasoning SFT (R0) Design** | Short CoTs, difficulty levels (L1–L3 shuffled), special reasoning tokens from unused vocab padding, logic-first taxonomy, and tool-use separation. |
| [Journal 12](journal12.md) | **100M/10B LR Schedule Debugging** | Investigating flat/diverging loss curves on 10B run, resuming step 15,500 on Beam, learning rate cooldown dynamics, and aggressive decay tuning. |
| [Journal 13](journal13.md) | **100M/10B Completion & Evaluation** | Wrapping up the 10B run across Kaggle/Beam/Modal, analyzing the flat loss tail, and benchmarking qualitative prose against GPT-2 Small (124M). |
| [Journal 14](journal14.md) | **Mixture-of-Experts (MoE) Foundations** | MoE motivation for scaling, top-$k$ routing, loss-free balancing with bias $b_i$, sigmoid & softplus routing, expert collapse, and auxiliary loss. |

---

## Detailed Breakdown

### [Journal 1: Dataset Selection & Sizing](journal1.md)
- Dealing with huge dataset sizes; streaming shards from HF vs local storage.
- Token budget sizing ($E = T / D$) and epochs.
- Comparing FineWeb-Edu, DCLM-baseline, Dolma 3, and Nemotron-ClimbMix.
- Strategy for code removal and category stratification.

### [Journal 2: Data Pipeline & Tokenization](journal2.md)
- Deciding against building a custom tokenizer from scratch; sticking with GPT-2 BPE tokens.
- Auditing Nemotron-ClimbMix cluster IDs.
- Streaming, buffering, and packing tokens into binary `.bin` files for training efficiency.

### [Journal 3: Architecture Exploration](journal3.md)
- Comparing Dense Transformers, MoEs, State Space Models (Mamba), and Linear Attention.
- Tied embeddings and their parameter/memory efficiency.
- Evolution of Linear Attention: standard KV memory, DeltaNets, Gated DeltaNets, Kimi Delta Attention (KDA), and Gated DeltaNet-2 (GDN-2).

### [Journal 4: Positional Encoding & Layer Components](journal4.md)
- Rotary Position Embeddings (RoPE) mechanics and why recurrent/memory layers skip RoPE.
- RMSNorm vs classic LayerNorm.
- Activation functions: ReLU vs GELU vs SiLU, Gated Linear Units (GLU), and SwiGLU FFNs.

### [Journal 5: Optimizers & Training Dynamics](journal5.md)
- SOTA optimizers: Muon (Momentum Orthogonalized Newton-Schulz) and SOAP (Shampoo in Eigenbasis).
- Intuition behind SVD and Muon orthogonalization; applying Muon per attention head.
- Parameter assignment split: Muon for 2D weight matrices, AdamW for embeddings, norms, and recurrent weights.

### [Journal 6: Pre-Training Qualification](journal6.md)
- Finalizing the 20.6M model geometry: 3 GDN-2 layers + 1 full gated attention layer.
- Packaging a clean 10M qualification dataset from Nemotron-ClimbMix.
- Verifying manifests, checksums, and token boundaries.
- Establishing the exact 306-update training plan and Kaggle T4 preflight harness.

### [Journal 7: First Run Live & SFT Roadmap](journal7.md)
- Watching loss decrease from 10.8 to 8.0 on the first live run; understanding perplexity ($e^{\text{loss}}$).
- Deciding to scale dataset size to 100M tokens before scaling model parameters.
- Chinchilla compute-optimal ratios (20 tokens/param).
- SFT stage 0 (S0) plan: conversational training with SmolTalk and ClimbMix regularization.

### [Journal 8: Speed Bottlenecks & Chunking](journal8.md)
- Diagnosing severe training speed collapse (3,000 $\to$ 400 tok/s).
- Mechanics of GDN-2 chunking (32-token chunks) and cumulative log-decay.
- Exploding exponentials ($e^{G_j - G_i}$) causing NaNs and triggering CPU chunk-halving loops.

### [Journal 9: FLA Kernel Optimization](journal9.md)
- Migrating to Fast Linear Attention (FLA) kernels: throughput leaps to 20,000 tok/s.
- Launching 2B token training on the 20M model; qualitative prompt evaluations.
- Technical post-mortem: eliminating CPU-GPU synchronous chunk-stepping with fused GPU Triton kernels and SRAM tiling.

### [Journal 10: 20M/2B vs 100M/2B & Sampling Behavior](journal10.md)
- Comparing 20M/2B and 100M/2B checkpoints; emergence of factual recall.
- Evaluating greedy sampling vs stochastic generation (temp=1, top-p=0.9, top-k=20) to kill repetition loops.
- Multi-GPU setup on Kaggle dual-T4 and gearing up for the 100M/10B run.

### [Journal 11: Reasoning SFT (R0) Design](journal11.md)
- Designing the R0 reasoning stage with short Chain-of-Thought (CoT).
- Difficulty levels (L1, L2, L3) shuffled rather than staged sequentially.
- Special tokens (`<reasoning>`, `<answer>`) allocated from unused GPT-2 embedding padding.
- Logic-first skill taxonomy (INF, DED, REL, CSP, IND, ABD, MAG) vs arithmetic computation.

### [Journal 12: 100M/10B LR Schedule Debugging](journal12.md)
- Diagnosing flat/stagnant loss curves on the 100M/10B training run.
- Resuming training from step 15,500 on Beam to test learning rate cooldown.
- Examining the impact of delayed vs aggressive learning rate decay policies.

### [Journal 13: 100M/10B Completion & Evaluation](journal13.md)
- Completing the 100M/10B run across Kaggle, Beam, and Modal accounts.
- Analyzing the flat loss tail and terminal LR impact.
- Qualitative comparison with GPT-2 Small (124M) on prose continuation prompts.

### [Journal 14: Mixture-of-Experts (MoE) Foundations](journal14.md)
- Motivation for MoE in low-compute, high-token scaling regimes.
- Router classification and top-$k$ expert routing.
- Routing stability techniques: loss-free balancing with bias $b_i$, sigmoid routing, and softplus routing.
- Expert collapse risks and the auxiliary load-balancing loss formulation ($f_i$, $P_i$, $L_{\text{balance}}$).
