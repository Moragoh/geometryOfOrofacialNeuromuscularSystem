# Text-Independent Triplet Training

## Goal
Learn an embedding `f(trial)` where trials from the same person are close and trials from different people are far, **regardless of which word/phoneme was spoken**. Verification = distance between two embeddings. Architecture and input representation (SPD vs raw) are still open — see `fingerprinting_plan.md`.

## Data (Experiment1 only)
- 12 subjects, **Voiced and Unvoiced** combined.
- Words: 36 classes × 10 trials × 2 manners = 720 trials per subject.
- Phonemes: 38 classes × 10 trials × 2 manners = 760 trials per subject.
- **Subject 11 has no phoneme data** (corrupted per the paper), so it only contributes words.

## Cross-validation
- 4 folds: **9 subjects train, 3 subjects test**, rotating so every subject is tested exactly once.
- Split is by subject — test subjects are never seen in training.
- No validation subject: train every fold for the same **fixed number of epochs** (chosen once in a pilot run).

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

## Testing
- For each fold, build pairs among the 3 test subjects only, same task only, with the same balance as training.
- Report **EER and ROC AUC** (threshold-free), separately for **word–word** and **phoneme–phoneme**.
- In the fold where Subject 11 is tested, it contributes word–word pairs only.
- Report per fold and averaged over the 4 folds.

## Caveats
- Each subject is a single recording session, so this cannot separate person identity from session/electrode placement. That needs multi-session recordings.
- Test pairs use words/phonemes seen in training (by other people), so this shows generalization to new people, not new vocabulary.

## Open decisions
- Model architecture and input representation.
- `P`, `K`, margin, mining strategy, number of epochs.
