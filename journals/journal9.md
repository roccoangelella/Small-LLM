Two days ago i started training and went to cinema. Odissey sucked. But on WandB i saw 20k token/s. It was amazing.
The new FLA GDN2 implementation brough training speed to a place i'd never though it would get. This speed is the result of using (i think) true production "attention kernels" rather than our past pytorch toy example.
I'd like to go as deep as i can on this, and, since training speed isn't appearently a problem anymore, I'll start a 2B tokens training on the same 20M model. I'll do this mainly because of two reasons:
1. It'll take not that much (roughly 30k updates)
2. 20M model trained on 500M tokens is starting providing meaningful asnwers, e.g:

**PROMPT:**
The Roman Republic was a period of ancient Roman civilization that began after

**CONTINUATION:**
vernal equinoxes, which were the first to be used as a form of the Roman Empire.

I mean, it doesn't make truly sense, but it's not just bullshit, right? Another cool asnwer is this:
**PROMPT:**
Alice: Did you remember to close the window?
Ben: I thought you had closed it.
Alice:

**CONTINUATION:**
 I was a little bit more concerned about the time it was on.
Ben: I was a little bit more concerned about the time it was on.

It can see Ben and Alice alternating!

And checkout this one:

**PROMPT:**
Text: I loved every minute of the film.
Sentiment: positive

Text: The plot was tedious and predictable.
Sentiment: negative

Text: The acting was excellent, although the ending was weak.
Sentiment:

**CONTINUATION:**
 negative

Text: The effect of the effect on the effect of the effect on the effect of the effect on the effect of the effect on the effect of

It sees patterns! Therefore keeping the dataset scale rise makes sense, and also no overfitting patterns are evident in graphs. Validation loss is still slowly decreasing at every update

Also this is amazing:
PROMPT:
Question: What is the capital of France?
Answer:

CONTINUATION:
 The capital of France is the capital of France.

---
Now, in the meanwhile that training starts, we'll dive a bit into the new GDN2 implementation, so that we know what our vibes produced.

### Our Kernels
A kernel is a function launched on the GPU and executed in parallel by many GPU threads. Cuda kernels are asynchronous, so the CPU can put stuff to do in the GPU queue without waiting for GPU operations to finish. A case in which the CPU has to stop and wait for the GPU results to make the next move is highly slow and undesired, because it breaks the asynchronicity feature that our cool gpu possesses. That's exactly what our GDN did, lol. Moreover, using very small kernels, therfore having more scheduling/launching operations, increases cpu work, slowing eveyrhting down. It's wise to use "big kernels" (actually, "better fused" rathern than just big) that do big maths inside the gpu instead of programming every sub operation. Of course, we used very small kernels. lol.

Examining the two "slowdown cases" separately, we had the first syncrhonous one in the AdaptiveChunkwiseGDN2Backend, responsible of shrinking the chunks where exponential decays became unhandable by GPU. This required a back and forth for between CPU and GPU where GPU would compute all the comumlateive decays and exponentials, filling the matrix, and CPU would be idle, waiting for the GPU output, deciding wether launching the next GPU task or shirnking the chunk size. This was the major performance blocker in our whole implementation. For every update, this meant 2048 tokens/32 chunks=64 chunks. Considering 6 GDN2 layers per every update, we had 64*6=384 chunk decisions for every microbatch. Since we had batch size=4, this meant 384*4=1536 decisions per update. Moreover, 1536 is the best case scenario: CPU saying "chunk size too big, halve it" means another gpu operation and another cpu check. This could grow a lot, reason why the tokens/sec metric slowly decreased as training proceeded and learned decay became increasingly stronger.

### FLA Kernels
FLA implementation doesn't care about checking scalars, halving chunk size,.. It keeps chunk size=64, always. Because of this constraint, FLA uses a  much smarter linear system solver that is tailored for a 64x64 matrix rather than our Pytorch general purpose triangular linear systems solver. 

Moreover, the 32 chunk loop is moved entirely on the GPU. No for loop inside the cpu that delivers the next task to the GPU iteratively. This is a massive improvement that removes a large slice of the cpu waiting time. I guess the truth is harder that how it sounds. Probably not something like loop.to("cuda").

A further improvement is that, instead of keeping tensors into VRam and moving them to the chip's shared memory only for the compute, and later moving them to VRam again, our FLA implementation keeps tensor tiles (explain the meaning) into the shared memory for the whole batch, writing only the finla result to VRam. that reduces back and forths between ram and shared memory (the GPU processors' one) and compute time therefore.

A third optimization that might be worth digging into is that FLA uses a completely custom backward pass, with an entirely dedicated Triton kernel that derives all the required differentiated quantities required for the update in an algorithmically "direct" way, using custom kernels instead of relying on granular PyTorch operations.
