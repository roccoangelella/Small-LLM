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
