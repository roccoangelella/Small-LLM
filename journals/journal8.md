The first engineereing problems comes out: we're getting increasingly lower speed in processing tokens during pre training. We start at roughly 3000 tok/s and after 2k updates our speed gets around 4-500 tok/s. That sucks. It'll take ages. To solve this problem we must grasp what's causing it.

We know that GDNs have internal memory: each new token causes some information to be forgotten and some new information to be written.

The dumbest way we could have implemented this is:
token 1 in
process memory
token 2 in
process memory
...

To exploit GPUs parallel computing, we applied a 32 tokens chunking: with a 2048 tokens context window, we run a sequence with 64 steps instead of 2048, reducing the "inter step" compute.

Now, decay is a learned param in GDN, ruling the models' forgetting about each channel (see journals for a deeper explanation).

Our chunkwise formula compares the **decay accumulated at different positions, exponentiating that difference**. Therefore, if two tokens have a decay difference of 40, these exponentials compound up to becoming either huge numbers or NaNs. That causes our computations to break. To solve this, we applied chunking reduction: shrink the chunk to shrink exponentials as well, but this means much slower compute.

Just for the sake of knowing what we vibecoded, the chunking algorithm uses log properties to reconstruct memory decay across tokens.

Since we're using chunking, we process tokens together: instead of multiplying state memory by the decay every time, we just multiply all the 32 tokens decays and get the "final forgetting". Instead of multiplying, we use logs: instead of 0.8*0.5*0.3 we use log(0.8)+log(0.5)+log(0.3): this sum keeps track of the total decay accumulated from step 1 to step 3, and the exponential of this sum is equivalent to the product we wrote before.

If we weren't applying chunking, we'd simply use only sequential operations just like any recurrent state: the state t would be influenced only by state t-1.

However, due to chunking (proof available, but i'll skip it) we need to keep track of the influence that every token t has on every next state t+i, therefore we track the decays occurred after token i and through every future one until token j as e^(G_j-G_i)

These exponentials end up in a triangular matrix that gets iteratively multiplied by the t states to run the GDN2 algorithm. When these exponentials explode, such matrix multiplication become unfeasible and we shrink the chunk size to ensure that we can run the algorithm for the whole chunk. 