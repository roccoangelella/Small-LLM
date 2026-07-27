And so it begins! The dataset will require us to run a nice stratification plus a fun detokenization, since appearently nvidia released only the dataset under gpt 2 tokens (basically a doc will be [2193,1293,829,43]. Funny.) Since we decided that we'll use our own tokenizer, we have to use tiktoken to bring back the dataset at its original status. We'd have to do that anyway if we want to run a real stratification.

Of course we won't run the whole tokenization on the whole dataset, we'll rather stream batches, detokenize, save up to we reache the desired size.

We found out that the Nvidia's paper and Hugginface dataset page are conflictual about the cluster IDs meaning. However it looks like papers' logic is the correct one.