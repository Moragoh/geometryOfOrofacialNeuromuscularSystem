# Text-Independent Triplet Training

## Goal
Demonstrate that orofacial EMG can be used for **authentication (verification)**: given two trials, decide whether they come from the same person, **regardless of which word/phoneme was spoken**. This is a feasibility demonstration, not a working system.

The model learns an embedding `f(trial)` where trials from the same person are close and trials from different people are far. YES/NO = distance between two embeddings compared to a threshold.

## Data (Experiment1 only)
- 12 subjects, **Voiced and Unvoiced** combined.
- Words: 36 classes × 10 trials × 2 manners = 720 trials per subject.
- Phonemes: 38 classes × 10 trials × 2 manners = 760 trials per subject.
- **Subject 11 has no phoneme data** (corrupted per the paper), so it only contributes words.

## Cross-validation
- 4 folds: **9 subjects train, 3 subjects test**, rotating so every subject is tested exactly once.
- **Split is by subject**, never by trial or pair — test subjects are never seen in training. This is what makes the "works on unseen people" claim hold.
- No validation subjects: train every fold for the same **fixed number of epochs** (chosen once in a pilot run).

## Architecture (proposed)
- Input: per-trial `22x22` covariance matrix, as in the existing pipeline.
- Embedding network: the existing SPDNet (`spdLearning/spdNet.py`: BiMap/ReEig ×3 → LogEig → 256-dim), with the final classification `Linear` replaced by a projection to an embedding (e.g. 64-dim), then L2-normalized.
- Same network (shared weights) applied to anchor, positive, and negative.
- Train with `MixOptimizer` (Stiefel weights break under plain `torch.optim`). Needs a new triplet training loop — `spdLearning/trainTest.py` assumes classification.

## Pairs and triplets
One model is trained on all trials. The only rule is how pairs are formed:

- **A pair is always the same task**: word–word or phoneme–phoneme. Never word–phoneme.
- Within a task, the two trials can be the same or different word/phoneme.
- Subject 11 simply appears in word–word pairs only. Because task never differs inside a pair, this does not leak.

A triplet is (anchor, positive, negative), all from the same task:
- **anchor**: person `p`
- **positive**: person `p`, any word/phoneme
- **negative**: a different person `q`, any word/phoneme

Balance so the model cannot use shortcuts instead of identity. For both positives and negatives:
- same word/phoneme 50%, different 50%
- same manner (Voiced/Unvoiced) 50%, different 50%

## Batching and loss
- Each batch: `P` people × `K` trials from one task, with the `K` trials spread across words/phonemes and both manners. Balance per person so Subject 11 is not under-represented.
- Triplet margin loss on L2-normalized embeddings: `max(0, d(a,p) − d(a,n) + margin)`.
- Mine semi-hard or batch-hard triplets within each batch; check mining does not break the balance above.

## Monitoring during training
No validation subjects, so these are diagnostics only:
- Training triplet loss (dropping to ~0 fast usually means triplets are too easy).
- Fraction of active triplets (loss > 0) per batch.
- Embedding spread (average pairwise distance) — heading to 0 means collapse.
- Training-pair EER.

## Evaluation
For each fold:
- Build balanced same-task YES/NO pairs among the **9 training subjects** and, separately, among the **3 test subjects**.
- Report on both train and test pairs, separately for **word–word** and **phoneme–phoneme**:
  - **EER** and **ROC AUC** (threshold-free, main metrics).
  - **Accuracy**, using the threshold chosen on the training pairs (e.g. the train EER threshold) applied unchanged to the test pairs.
- In the fold where Subject 11 is tested, it contributes word–word pairs only.
- Report per fold and mean ± spread over the 4 folds.

A small train/test gap shows the model learned something general about people rather than memorizing the 9 training subjects.

## Baselines
Same folds and same test pairs for all:
1. **No training**: log-Cholesky geodesic distance between covariance matrices (`basicOperations/manifoldOperations.py` `matrixDistance`).
2. **Classification-trained SPDNet**: train a 9-way subject classifier with the existing loop, use the LogEig output as the embedding.
3. **Triplet-trained SPDNet**: the main model.

## Caveats
- All recordings are single-session, so this cannot separate person identity from session/electrode placement. Cross-session robustness is left to future work.
- Test pairs use words/phonemes seen in training (by other people), so this shows generalization to new people, not new vocabulary.

## Open decisions
- Embedding dimension, `P`, `K`, margin, mining strategy, number of epochs.
