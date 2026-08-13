Two major improvements have been achieved in those days thanks to the 20M/2B and 100M/2B runs.
In both, the models start giving correct answers e.g. Recognizing that the capital of France is Paris, that Jupyter is the largest planet in the solar system. 
In particular, the 100M model shows a significantly better capability of continuing the text after the continuation is provided. 

The models tets have been obviously carried on using the same temperature, top k, top p params, using a fully deterministic model (temp=0, top p=1, top k=0). Now i'm running another test suite using temperature=1, top p=0.9, top k=20 to see how a more stochastic model behaves. I'm doing this because i feel like the greedy sampling is mainly responsible of "endless repetitions" after the true answer has been provided by the model. For example, after correctly responding "The Pacific Ocean" to the question "What's the largest ocean on Earth?", the 100M 2B model would simply repeat "Question: What is the name of the largest ocean on Earth? Answer: The Pacific Ocean". I would like to find out wether this repetition is caused by the model being forced of greedily choosing the most likely token. Gonna be a fun comparison.

In the meanwhile, i'm wiring the next test, staying consistent with the "scale data until no more improvement is on sight" principle: the next run is gonna be 100M/10B. Observing training graphs, there's clearly much room for validation improvement. Maybe not enough to justify 5x more tokens, but I mean we're not paying for GPUs are we?
![alt text](image.png)


Speaking about GPUs, we wired Kaggle training to use both T4. We can forget 60k tokens/second that our Modal's H100 achieves on the 100M model, but it's gonna be a nice plan b. 
