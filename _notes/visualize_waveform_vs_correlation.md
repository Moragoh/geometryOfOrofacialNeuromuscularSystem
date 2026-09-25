# Context

This repo is about founding out what data it is better to train the model on--raw waveform or correlation matrices.

For repetitions of the same phoneme/letter in the same manner, I want to visualize waveforms side by side and correlation matrices (after they have been processing following how the repo processes them). My hypothesis is that waveforms of repetitions will look different, while correlation matrices of different repetitions will still look similar enough. D

# Waveforms:

I want one channel visualized at a time: One repetition on top, the other below, then onto the next channel

# Correlation MatriX

What would be the best way to visualize these?

# Notebook

Create a notebook. Wheneber it starts, it should pick two samples that are repetitions of each other (same subject, manner, phoneme/word) then it should visualize. Every time it runs, it should visualize a new pair.

Put this is directory \_visualize_data/ at the root/
