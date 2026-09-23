# General notes for the pipeline

These are notes from a human trying to understand this pipeline for the purposes of repurposing it.

- Let's use experiment 1 data for now.

## Experiment 1

- 12 subjects
- 13 facial movements, 38 phenomes, 36 words (Use only phenomes and words)
- Voiced/Unvoiced for phenomes and words
- 10 reps per class

- 2 models are trained on Experiment 1 data (only phonemes and words)
- SPDNet: Takes in covar matrix of one trial (one word of phoneme), tries to predict which word/phoneme it was.
  - Why covar matrix and not raw data?

- SPD-RNN: One trial is represented as a sequence of 46 covariance matrices, and the model predicts which word/phenome it was.
- SPD-RNN is able to look at the phenome/word over time (such as which muscles move first before others). SPDNet squashes all that into one covariance matrix, which represents which muscles are involved. Two words can use the same muslces but in different order. SPD-RNN addresses that.

## Training

- Decided on pair generation scheme.
- Need to decide spd or raw data
  - Need to learn how to shape data from preprocessd -> SPD
  - Then once we get to SPD data, we think about how to generate pairs
- To do this, I need to understand the spd-rnn pipeline (learn spd transformation)
- Write training model.

- Understand:
    - How data is transformed to SPD
    - How to generate pairs.
    - How the model training works.