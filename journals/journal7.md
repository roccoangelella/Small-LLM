Aaaaand we are live!

The first pre training process is currently (quite smoothly though!) flowing from a 10.8 to a 8.0 cross entropy loss during the first 50 updates. Validation loss is 7.91 at this stage. Nice! Perplexity is 2739.35 at this point.

What the hell is perplexity? I always asked it myself! It's just the exponential of the cross entropy loss. It's interpretation is, roughly: 
**How many tokens is my model scanning through, to choose the correct one?** very simply, if the model has loss=0, e^0=1, meaning it is considering only one token, the correct one.

When we had loss 7.91, our model was uncertain about 2739.35 tokens. Plain and simple. Considering that the model has to pick among roughly 50k tokens, that's still something.

I noticed we're using only 32% of GPU during this run. Might be useful to try microbatch=2,3,4,5... in next runs.