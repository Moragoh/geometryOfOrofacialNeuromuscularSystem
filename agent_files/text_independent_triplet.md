# Text-Independent Triplet Training

## Goal
Learn an embedding `f(trial)` where trials from the same person are close and trials from different people are far, **regardless of which word/phoneme was spoken**. Verification = distance between embeddings below a threshold. Architecture and input representation (SPD vs raw) are still open — see `fingerprinting_plan.md`.

## Training data (Experiment1)
- Use **Unvoiced** only, to match Experiment2 (NATO codes were articulated silently).
- Words (36 classes) and Phonemes (38 classes; Subject 11 missing) — 10 trials per class per subject.
- Split by **subject**, not by trial: 10 subjects for training, 2 held out for validation.

## Building triplets
A triplet is (anchor, positive, negative):
- **anchor**: person `p`, word `w`
- **positive**: person `p`, any word
- **negative**: a different person `q`, any word

The key constraint: **whether the word matches must not predict same/different person.** Balance it so that:
- positives are the same word 50% of the time, a different word 50%
- negatives are the same word 50% of the time, a different word 50%

If positives were always different words and negatives always the same word (or vice versa), the model could learn word matching instead of identity.

## Batching and loss
- PK sampling: each batch has `P` people × `K` trials, with the `K` trials spread across different words so both same-word and different-word pairs exist within and across people.
- Standard triplet margin loss on L2-normalized embeddings: `max(0, d(a,p) − d(a,n) + margin)`.
- Mine semi-hard or batch-hard triplets within each batch. Check that mining doesn't break the word balance above (e.g. hardest negatives all turning out to be same-word).

## Validation
- On the 2 held-out Experiment1 subjects, score balanced genuine/impostor pairs and compute EER.
- Use it for early stopping and to **fix the decision threshold before touching Experiment2**.

## Evaluation (Experiment2 — 4 unseen people, unseen words)
- **Enroll**: for each person, one template = mean embedding over all of `trainSet.npy` (26 codes × 20 reps, ordered by code).
- **Probe**: each trial in `rainbowPassage.npy` / `grandfatherPassage.npy` (labels in `...Labels.npy`) is compared to all 4 templates → 1 genuine score, 3 impostor scores.
- Report EER and ROC AUC, overall and per person.
- For comparison, also run the text-dependent version (per-code templates, compare only to the probe's code). The gap shows how much the fingerprint depends on knowing the word.
- Passages were recorded after the full `trainSet`, so this also tests within-session drift.

## Caveats
- Only 4 test identities (6 impostor person-pairs) — treat results as a demonstration, not an estimate.
- Each person is a single session, so this cannot separate person identity from session/electrode placement. That needs multi-session recordings.
- Any setup difference between Experiment1 and Experiment2 shifts all test people together; if performance collapses, try re-centering each experiment by its overall mean before embedding.

## Open decisions
- Model architecture and input representation.
- `P`, `K`, margin, and mining strategy.
- Whether to add Voiced trials later as extra positives (same person, different articulation manner).
