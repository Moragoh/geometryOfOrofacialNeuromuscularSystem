# Context
Read @gowda.pdf for context on the paper this repository was used to write.

# Overview
I plan to reuse parts of this repository and how it trained models on EMG, but for a model that performs fingerprinting.

# Fingerprinting
The current repository asks if EMG can be used to infer what phenomes/words a person is speaking to allow for speech prosthetics. This is about which phenomes are being spoken within the same person. We ask a different question: can EMG for these individuals speaking phenomes/words be used to fingerprint them?

This is the core idea of how I imagine this will work: the dataset has each participant speak different classes of words with 10 repetitions each. We are now no longer trying to see if we can differentiate between phenomes/words given all the individuals. We are asking if we can differentiate between people given all phenomes/words.

This is what I imagine one data sample to be:
EMG data of the same person speaking the same phenome/word => model is supposed to answer YES
EMG data of a different person speaking the same phenome/word => model is supposed to answer NO

It is important that the two samples the model is being asked to say YES/NO is of the two (or the same individual) saying the same class. If those were different, then the model could confound the word/phoneme being different with the person being different. Does this make sense? Tell me if there is a better way.

# Open Questions
- Should we try training on the raw emg data per trial rather than SPD?
- If we use SPD, should we use spdnet or spd-rnn?
- Do you think triplet loss would be the best way to train this model?