I'm now starting to design what comes after the conversational SFT: a second SFT stage focused specifically on reasoning.

At first my idea was quite simple: teach the model short CoTs, split them in three levels of toughness, and gradually make the reasoning harder. Turns out the first part seems wise, while the second one isn't really supported by recent papers. Easy-to-hard ordering doesn't appear to give a consistent advantage, especially on small models. So we'll still have three levels, but we'll **shuffle them together during training** instead of feeding the model an increasing challenge.

The three levels will basically describe how many dependent reasoning steps are needed:
1. **L1**: one atomic inference.
2. **L2**: roughly 2-3 dependent inferences.
3. **L3**: roughly 3-5 dependent inferences, eventually with a little branching/elimination.

The labels will NOT be shown to the model and they won't affect sampling. They'll just stay in the dataset metadata. This is actually pretty interesting because during validation we can separately track L1/L2/L3 loss and see how the model learns each toughness level. Even better, we'll also track the reasoning skill, so we'll basically have a skill x difficulty matrix and see where the 100M model sucks the most while training. Nice.

Another important decision concerns how the reasoning itself is serialized. I like the idea of having three special tokens to distinguish reasoning from the actual answer, with the model being free to either think or answer directly. Something conceptually like:

<reasoning>
...
</reasoning>
<answer>
...

The exact strings aren't frozen yet. The important thing is that they're atomic tokens. Funny enough, our GPT-2 vocab has 50,257 real tokens but our embedding matrix has 50,304 rows because we padded it for hardware alignment. Therefore we already physically have 47 unused embedding vectors. We can simply promote three of them to semantic tokens, initialize them using the same Normal(0, 0.02) initialization used for the other embeddings, and train them during the reasoning SFT. No matrix resizing and no new params. Pretty lucky accident lol.

We'll still run a tiny ablation comparing normal text markers like "Reasoning:" / "Answer:" against the three special tokens. I expect we'll use special tokens anyway, but since the test is cheap there's no reason to vibe-decide it.

The model deciding wether to reason is also important because i don't want to teach it that every stupid question needs a CoT. Ideally something obvious can go directly to <answer>, while a prompt requiring a few inferences can enter reasoning mode. SFT will mainly teach that this reasoning mode exists and how to produce short valid reasoning. Later RL should become the step that teaches **when** reasoning is worth using and **how much** of it is useful. I initially thought about simply penalizing reasoning length, but recent work suggests that blindly doing that can teach the model to stop thinking too early. So later RLVR should keep correctness primary and only reward shortness among already-correct reasoning paths.

---

Then came the dataset question. We could hunt for some premade CoT dataset, but since i wired a Gemini free-tier wrapper and we have several hundreds of calls available, we'll generate the R0 dataset ourselves. A single call can generate many examples, so several hundred calls should already be enough to produce a few thousand candidates.

The important part is that Gemini shouldn't decide everything. We'll define the requested skill and toughness ourselves, ask it for concise structured examples, and locally verify/reject the outputs. Therefore the final architecture is roughly:

our generator -> choose skill + L1/L2/L3 -> Gemini generates/naturalizes the task and short trace -> local verifier -> accept/reject -> frozen dataset.

This gives us much more control than simply downloading a huge reasoning dataset made for a 7B or 30B model.

---

The biggest change of mind was about **what reasoning we actually want to teach**.

My first thought was that reasoning datasets usually contain loads of arithmetic and word problems, but then i started questioning wether it makes any sense to spend the limited capacity of a 100M model trying to make it calculate. Exact multiplication, percentages, algebra and calculus feel like something that a calculator/tool should simply do better. What i'd rather have inside the model is the logic required to understand *what needs to be calculated*, and basic numerical awareness such as knowing that $7 < 12$, $-3 > -7$, what a range means, which quantity is larger, etc.

Recent papers seem to support this distinction much more than i expected. They explicitly separate mathematical reasoning from arithmetic computation, while logical primitives like deduction, induction, abduction, quantifiers and relational reasoning show meaningful transfer. Small models can actually learn arithmetic better than i thought, so "LLMs can't do maths" is clearly too strong, but that doesn't mean it's the best use of our tiny model's capacity.

Therefore **R0 will be logic-first**, with the following skill taxonomy:

- **INF**: immediate inference, negation, quantifiers, implication, AND/OR.
- **DED**: deduction from explicit rules and premises.
- **REL**: relations, ordering, transitivity, containment, temporal/spatial relations.
- **CSP**: satisfying multiple constraints and eliminating impossible options.
- **IND**: induction, deriving a compact rule from observations.
- **ABD**: abduction, choosing the explanation that best fits the supplied evidence.
- **MAG**: numerical magnitude awareness: $<$, $>$, $=$, sign, ranges, min/max and approximate magnitude.

No calculus. No goal of turning the model into a calculator. Exact computation will come later through tool-use SFT + tool-aided RLVR. The model should learn the procedure and logic, the tool should execute the exact maths.

I actually like this architecture a lot because it also feels closer to how humans learn. We don't start by staring at thousands of multiplications. First we learn that numbers represent quantities, that one can be larger or smaller than another, that things follow logical relations, and only later we learn operations over those representations. Of course neural networks don't learn exactly like children do, but as an intuition this separation between **representation/logic first and operations/tools later** makes much more sense to me than asking a 100M model to memorize being a calculator.

So the current post-training idea is becoming:

conversational SFT -> short logic-oriented R-SFT -> tool-use training / RLVR for computation -> preference/alignment stages later.

Let's see if our little guy can actually learn to think :)
