# Purpose

Test the hypothesis: **given enough data, a model trained on the raw waveform does not need SPD matrices.**

The paper turns each trial into covariance matrices before training. Covariance squares the signal and averages it over a time window, which throws away the random fine timing of motor-unit firing and keeps only how strongly (and how jointly) the channels activated. The question is whether a neural network can learn to extract that on its own, or whether the SPD preprocessing is necessary.

To test that, we need a model that learns from the raw waveform **without clever tricks that reshape the data for it**. If the model has covariance-like operations built in (explicit squaring, pooling over a hand-picked window, etc.), it is just doing the SPD preprocessing inside the network, and we might as well have used the covariance matrices. That is why Conv+GRU is not a good fit: its downsampling already makes it look at the signal at a coarser time scale, which is partly what covariance does.

# Model: ResNet1D

We use the standard ResNet for time-series classification from Wang et al. 2017 ("Time series classification from scratch with deep neural networks"), unchanged.

## Why this model

- **It only assumes things that apply to any time series.** Nearby samples are related (locality), and a pattern means the same thing wherever it occurs in time (shift equivariance). Nothing in it is specific to EMG.
- **No hand-picked time scale.** Each conv layer only sees a few milliseconds of signal. Any coarser structure has to be learned.
- **No squaring or covariance-like operations.** If the network ends up computing something like signal energy, it learned to do that itself.
- **It is the field's default, not our design.** It was the strongest model in the Fawaz et al. 2019 time-series classification benchmark. Using it as-is means we can't have (even accidentally) built the answer into the architecture.

## Why not something even less structured

- **MLP on the flattened input:** can't even tell that a pattern at 200 ms and the same pattern at 800 ms are the same thing, so it would fail at any realistic data size and tell us nothing.
- **Transformer:** even fewer built-in assumptions, but known to need a lot of data. With 6 repetitions per class it would almost certainly fail, which wouldn't tell us much beyond "not enough data."

## Architecture

- **Input:** `(22 channels, 7500 samples)` per trial, z-scored per trial and per channel over time, the same as the SPD notebooks. This keeps normalization identical across models, so any accuracy difference comes from the architecture alone. (Train-set channel normalization, which would keep amplitude information, is deferred.)
- **3 residual blocks.** Each block:
  - Conv1D (kernel 8) → BatchNorm → ReLU
  - Conv1D (kernel 5) → BatchNorm → ReLU
  - Conv1D (kernel 3) → BatchNorm
  - Shortcut: 1x1 Conv1D + BatchNorm when the channel count changes, otherwise BatchNorm only; added to the block output, then ReLU
- **Filters per block:** 64 → 128 → 128
- **Global average pooling** over time → **Linear** layer to the number of classes (38 phonemes or 36 words)

Everything else stays the same as the SPD experiments: Subject 1, voiced/unvoiced, all phonemes/all words, and the same train/test split (trials 0–2 and 5–7 train, trials 3–4 and 8–9 test).

# High level plan

## Normalization

- Original paper: applies (x-mean)/std per channel, per trial. This means that every data sample reflects how much it differs from the mean of itself. It preserves the difference, but the raw amplitude information is thrown away. So there is no scale difference between different trials. There is only "how much this sample is different from its own average" per sample. So across subjects there is: "How much this person moves a muscle more in this sample relative to themselves", but not "How much more this person moves their muscle compared to another person."

This method of zscoring allows the SPD matrices to becomne a correlation matrix, which only looks at: "How did these channels coordinate." If we do not do it, we just get a covariance matrix. A large covariance cannot answer between: "low amplitudes but highly coupled" vs "high amplitudes, loosly coupled"
cov(a, b) = corr(a, b) × std_a × std_b
But z scored per trial, std_a = std_b = 1

(What is thrown away): Relative amplitude diffs between trials, '' between channels.
(What is kept): Within a trial, how much a channel covaried with one another.

- What normalization we will use
  What information I don't want to throw away: between words and phonemes, maybe some words just recruit stronger responses. After all, harsher sounds use more recruitment of the muscles, and that is a valid piece of information that can be used to differentiate between sounds. So maybe we zscore per trial, but not per channel. This allows for magnitude differences between channels to be preserved while normalizing and gettig rid of per trial differences.

So we could be calculating mean and std: per channel, across all training trials.
This allows us to normalize and keep things at the same scale, while also getting rid of electrode gain differences: Since normalization is done per channel, no amplitude difference between channels survive1.

However, we calcualte mean/std across trials, which means that differences between trials remain. This means that how much a channel activates compared to another class survivies. The downside is that relative channel differences between repetitions survive

Recap:
(What gets thrown away): Relative amplitude differences between channels in one trial.
(What is kept): Relative amplitude differences between classes AND relative amplitude diffs between repetions of same classes.

What we ended up doing:
zscoring same as pipleine.

## resnet1d overview

It has three residual blocks stacked together.
Each residual block is conv1d layers + batchnorm + the shortcut that adds the original input (because this is resnet and adding this is what prevents vanishing gradients). So the shortcut is added once per block before its output.
