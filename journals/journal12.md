The 100M/10B run is quite disappointing:
![alt text](image-1.png)

I mean, how's that even possible? The most likely answer is how we handle learning rate. A similar behavior wasreported during the 20M run, comparing 2B vs 500M:

![alt text](image-2.png)

To investigate this phenomenon, we're gonna run a small test: take the 100M snapshot at step 12500 (that's where we clearly see 2B and 10B runs performances deviating) and resume the training from that point with a heavier lr decaying: in both 100M/2B and 20M/500M runs, the heavy decrease in val loss is caused by triggering of heavy lr decay. It looks we misprogrammed LR scheduling to happen correctly also on longer runs.

So, the experiment will involve using the same 2B run decay on the 10B run. If loss goes down, we know that's the reason.