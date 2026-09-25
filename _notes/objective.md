# Objective

The current objective is to recreate B.2.2. and B.2.3. results, but on the raw data rather than the SPD matrices.

# Background

The original paper @gowda.pdf has a part where models are trained per participant to classify which phoneme or class is being said.

## Data

Phoneme data:

- 11 participants (subject 11 phoneme data corrupted)
- 38 phonemes, (38 spans the entire Enlgish language space)
  - 23 consonants, 15 vowels.
  - repeated 10 times voiced, 10 times unvoiced

Word data:

- 12 participants
- 36 words (spansned the entire English language phonemic space)
  - repeated 10 times voiced, 10 times unvoiced.

# Results I am trying to recreate

They created SPD representations from this word/phoneme data (either one matrix for SPDNet or a sequence of 46 matrices of SPD-RNN). Then they trained a classifier on it. Models were trained per subject. They ran multiple experiemnts (such as the NATO alphabet spelling experiment), but the focus of this project currently is to simpyl recreate the word/phoneme classification.

- Before feeding the raw signals into the model, the data is turned into SPD matrices. The authors train 2 models: SPDNet and SPD-RNN. Processing from raw signal to SPD matrices differ based on what model it is:
  - SPDNet: each trial gets transformed into a 22x22 covariance matrix.
  - SPD-RNN: EMG signal (1500ms) gets split into slices of size 150ms at step 30ms, resulting in 46 covariances.

- Train/test split (for both phonemes and words)
  - Out of 10 total repetitions, 6 are used for train, 4 are used for test.

- Model variations
  - Models are trained per person, task, manner (silent or voiced)
    - There are 4 tasks: all phonemes, consonants phonemes only, vowel phonemes only, all words.
  - For our purposes, lets compare results for the all phoneme model and the all word model.

## Final recap of what specifics from the paper I want to recreate

### What the paper does

I want to recreate the results that look into how well neural networks can classify phonemes and words. This would be the experiments described in Section 2.3.6 and Section 2.3.7. I need to recreate the results and compare them against Table 8-11. I am telling you this limited scope now so you know exactly what we are trying to replicate; we are not trying to replicate all the results; only some.

This is what the paper does:

1. Take raw signal from the dataset
2. Zscore each column per trial so that the units are consistent across channels and trials.
3. Create covariance matrix.
   - If using SPDNet, one trial => one 22x22 matrix
   - If using SPD-RNN, one trial => 46 22x22 matrix
4. Feed matrices into SPDNet or SPD-RNN based on what format it was converted to.
5. Model classifies what phoneme/word it is.

The paper trains a separate model per:

- Participant
- Manner: voiced/unvoiced
- Task: all phonemes, consonants, vowels, all words (we know this because the paper cites separate accuracies + notebooks exist for each)

# Training specifices:

Within each model configuration (participant/manner/task), the task is now to recognize which class (which phoneme or word was said).

Each class has 10 repetitions (trials). The paper splits each class the same fixed way:

- 6 for training (trials 0–2 and 5–7) # This split trial split is done in the code.
- 4 for testing (trials 3–4 and 8–9)
- NOTE: To ensure that the accuracies are the same, we must use the same indices for training and testing as the codebase does.
- NOTE 2: So there is very little data. The model only gets to see 6 examples for each class.

So the model takes in the matrices for the 6 training samples per class and learns on: which class is this?
Then it is tested on the remaining four per class and is asked: which class is this?

# What I want to do
I want to recreate this repo's results for these specific metrics:
- Participant 1, voiced, all phonemes
- Participant 1, unvoiced, all phonemes
- Participant 1, voiced, all words
- Participant 1, unvoiced, all words
And see the test set accuracy for classifying which phoneme/word it was

THE CATCH: Instead of using SPD matrices, I want to use the raw signals and compare perfomance.
The main goal is this: is it better to use SPD matrices as training data, or raw data?

## Parameters that must match
In order to isolate any performance difference to what form of data was used, some parameters must be fixed.
- Same participant/manner/task type model and the appropriate data
- Same train/test split (same repeition numbs used as train/test)
- Raw model: no per-trial z-scoring; each channel is normalized with a mean/std computed once over the training set

## Open Questions
- Obviouly the model architecture will be different. What architecture would be the best?

# Extras I need to know

## What is EMG?

When muscles fire, its fibres produce small electrical impusles that can be measured as voltage on the skin surface. The stronger a muscle fires, the stronger the EMG reading.

## Why does the paper love covariance matrices when then raw data has 22 channels?

- Raw waveform:
  - When a muscle fires, many motor units activate at uncontrollable times. So for the same muscle movement at the same strength, the timing of the voltage can look very different. This leads to waveforms being very messy for the same words.
  - If the timing differs, howcome the downstream muscle action is the same?
    So at a high level: minute timing diffs doesn't matter for the muscle movement because it takes the overall activation and intreprets that. But the EMG channels pick up all those minute differences, so the raw waveform has random timing differences fort the same action.

- Covariance matrix:
  - A covariance matrix measures coactivations between channels within a time window (150ms for RNN, 1.5 seconds for SPD). Wihtin that time window, however, the exact timing of the electrical currents do not matter. Only how much it activated inside that window overall does. This dampens the randomness from "motor muscle fires at different times."
  - When covariance matrices are calculated for a time span, the time dimension is thrown away as all the channel values across t is summed up. This is why SPDNet data divides the matrix into 7500. To get an average. So this averaging makes sure we are only measuring activation amount.
  - Covariance matrices are purposely about overall activation, not timing. The RNN gets over this by providing a coarse grain look at time, but the time information within slice is thrown away. The waveform on the other hand has fine grain timing information and none of it is thrown away, so t can become a very distracting data stream (because t by human biology isna very random thing). Covariance matrices make the job easier on sparser data by throwing this data away for the model.

- Overall: same actions have very different voltage timings due to how motor units are random. Covariance matrices by nature throw away lots of timing data, so it reduces noise for the model. If the model had enough data, maybe the covariance matrix wouldn't be necessary, but we only have 6 samples per class.

