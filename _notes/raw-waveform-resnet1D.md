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

- **Input:** `(22 channels, 7500 samples)` per trial. No per-trial z-scoring (that was part of the SPD machinery). Instead, each channel is normalized with a single mean/std computed over the training set and applied to every trial, train and test. This only fixes differences in electrode gain and overall scale so the network trains reliably, and keeps how loud each channel was in each trial.
- **3 residual blocks.** Each block:
  - Conv1D (kernel 8) → BatchNorm → ReLU
  - Conv1D (kernel 5) → BatchNorm → ReLU
  - Conv1D (kernel 3) → BatchNorm
  - Shortcut: 1x1 Conv1D + BatchNorm when the channel count changes, otherwise BatchNorm only; added to the block output, then ReLU
- **Filters per block:** 64 → 128 → 128
- **Global average pooling** over time → **Linear** layer to the number of classes (38 phonemes or 36 words)

Everything else stays the same as the SPD experiments: Subject 1, voiced/unvoiced, all phonemes/all words, and the same train/test split (trials 0–2 and 5–7 train, trials 3–4 and 8–9 test).
