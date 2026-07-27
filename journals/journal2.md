And so it begins! The dataset will require us to run a nice stratification plus a fun detokenization, since appearently nvidia released only the dataset under gpt 2 tokens (basically a doc will be [2193,1293,829,43]. Funny.) Since we decided that we'll use our own tokenizer, we have to use tiktoken to bring back the dataset at its original status. We'd have to do that anyway if we want to run a real stratification.

Of course we won't run the whole tokenization on the whole dataset, we'll rather stream batches, detokenize, save up to we reache the desired size.

We found out that the Nvidia's paper and Hugginface dataset page are conflictual about the cluster IDs meaning. However it looks like papers' logic is the correct one.

Running some tests (that kimi invented and I didn't fully understood), i got this conclusion. We can approximately trust cluster IDs and use them for a full production download of the dataset. Not fully though, since we saw that not all cluster IDs perfectly match the topic, but they match enough for us to trust them for at least this first step. We'll be free to retry with deper testing if this doesn't work.

Some open questions:
- How should i divide the % of each cluster? Simply 5% for each of the 20 IDs?
- Which format will the dataset be? Parquet? Json? 