# Dataset Generation Plan (Triplet Fingerprinting)

High-level plan for turning the Experiment1 data into (1) per-trial SPD matrices, (2) metadata for pair/triplet generation, (3) fold assignments, and (4) fixed evaluation pairs. Training triplets are generated on the fly from (1)–(3). See `text_independent_triplet.md` for the training/evaluation design.

**Consistency rule:** SPD matrices must be produced with the exact same code the repository used for the words/phoneme results in the paper. Do not re-implement or "improve" it.

## Prerequisites: read these first

| File | Where | What it shows |
|---|---|---|
| `wordsSPDNet.ipynb` | cell 5 (constants), cell 7 | **Canonical SPD pipeline for words**: load → per-channel z-score → class labels → `1/windowLength * X @ X.T` |
| `allPhonemesSPDNet.ipynb` | cell 1 (docstring), cell 5, cell 7 | Same pipeline for phonemes; docstring describes array layout `(380, 22, 7500)` and phoneme label order |
| `wordsMDM.ipynb` | cells 5, 7 | Same pipeline looped over all subjects, for Voiced (cell 5) and Unvoiced (cell 7); subject list includes Subject 11 |
| `phonemesMDM.ipynb` | cells 5, 7, 13, 15 | Same pipeline for phonemes; subject list excludes Subject 11. Cells 13/15 index vowels with a `230 +` offset, confirming trials are stored class-major with the 23 consonants first |
| `orofacialTsne.ipynb` | cell 4 | A **different** variant (`normalize(...)` + `0.9 * E + 0.1 * trace(E) * I`, matching the paper's η = 0.1 shrinkage). Used only for orofacial movements — **not** used for words/phonemes. Read it to understand the discrepancy, but do not use it |
| `spdLearning/spdNet.py` | `learnSPDMatrices` | Model input is `(batch, 22, 22)`; it adds the channel dim itself via `unsqueeze(1)` |
| `wordsSPDNet.ipynb` | cell 3 (`BaseDataset`) | Matrices are stored as float64 and cast to float32 only when served to the model |
| `gowda.pdf` | §2.2, §2.3 | Paper's description of preprocessing and edge (covariance) matrices |

### The exact repo pipeline (from `wordsSPDNet.ipynb` cell 7)
```python
DATA = np.load("Experiment1/Words/" + articulationManner + subject + ".npy")   # (360, 22, 7500)
mean = np.mean(DATA, axis = -1)
std = np.std(DATA, axis = -1)
DATA = (DATA - mean[..., np.newaxis])/(std[..., np.newaxis] + 1e-5)
# per trial:
covariance = 1/windowLength * (DATA[trial] @ DATA[trial].T)                    # windowLength = 7500
```
No shrinkage/regularization is applied in the words or phoneme notebooks, even though the paper's §2.3 describes η = 0.1. Follow the notebooks, since they produced the reported words/phoneme numbers.

## Step 1: Inventory and sanity checks
- For each of the 12 subjects, locate:
  - `Experiment1/Words/{Voiced,Unvoiced}Subject<N>.npy` — expect `(360, 22, 7500)`
  - `Experiment1/Phoneme/{Voiced,Unvoiced}Subject<N>.npy` — expect `(380, 22, 7500)`
- Confirm Subject 11 phoneme files are missing or unusable (paper: "corrupted and unavailable"). If present, exclude them anyway.
- Check for NaNs/Infs and flat channels.

## Step 2: Generate SPD matrices
**Reuse:** the load + z-score + covariance lines from `wordsSPDNet.ipynb` cell 7 (words) and `allPhonemesSPDNet.ipynb` cell 7 (phonemes), unchanged — same `axis = -1` statistics, same `+ 1e-5`, same `1/windowLength` scaling.

- Loop over subject × task (Words, Phoneme) × manner (Voiced, Unvoiced), skipping Subject 11 phonemes.
- Compute one `22x22` matrix per trial.
- **Do not** carry over the rest of cell 7 (the `0:3 + 5:8` / `3:5 + 8:10` train/test split). That split is for word classification within one subject; our split is by subject.
- Sanity check: every matrix is symmetric with smallest eigenvalue > 0.

## Step 3: Build per-trial metadata
One row per SPD matrix:

| column | source |
|---|---|
| `covariance_index` | row position in the saved array |
| `subject` | file name |
| `task` | `word` / `phoneme` (directory) |
| `manner` | `voiced` / `unvoiced` (file name prefix) |
| `class_id` | position in file, using the repo's labeling: `labelsByWords = np.array([[i] * 10 for i in range(numberClasses)]).reshape(numberTrials)` (class-major; class `i` = trials `10i … 10i+9`) |
| `repetition` | index within the class (0–9) |

Note: the phoneme docstring in `allPhonemesSPDNet.ipynb` lists 36 names for 38 classes (AW and AY are missing; the paper lists all 38). This only affects human-readable names, not `class_id`.

## Step 4: Verify consistency with the paper
Before using the saved matrices for anything new, run the existing `wordsMDM.ipynb` logic on the saved covariances for a couple of subjects and confirm the per-subject accuracies match the paper's Table 7 (e.g. Subject 1 audible words 0.632). This confirms the SPD generation and `class_id` ordering are identical to the paper's.

## Step 5: Save
One file (e.g. `.npz`) containing:
- `covariances`: `(N, 22, 22)`, float64 (cast to float32 at load, as `BaseDataset` does)
- the metadata columns from Step 3

Size is small (~17k matrices, ~35 MB).

## Step 6: Fold assignment
- Split the 12 subjects into 4 groups of 3 with a fixed seed; save the assignment.
- Fold `k`: test = group `k`, train = the other 9 subjects.
- Record which fold tests Subject 11 (word pairs only in that fold).

## Step 7: Training triplets (on the fly, not saved)
A sampler reads the metadata and builds each batch:
1. Pick one task. Phoneme batches exclude Subject 11.
2. Pick `P` training subjects and a small set of shared classes.
3. Take trials of those classes from each subject in both manners.
4. For each anchor, draw:
   - **positive**: same subject, different trial; 50% same class / 50% different; 50% same manner / 50% different
   - **negative**: different subject; 50% same class / 50% different; 50% same manner / 50% different

## Step 8: Evaluation pairs (fixed, saved)
Per fold, with a fixed seed:
- Build balanced same-task YES/NO pairs among the 9 training subjects and, separately, among the 3 test subjects.
- Balance each label across same/different class and same/different manner, as in Step 7.
- Save as `(covariance_index_a, covariance_index_b, same_person, task)` so every model is scored on identical pairs.

## Outputs
- `covariances` + metadata file
- fold assignment file
- evaluation pair files (per fold, train and test)
- the triplet sampler (code, reads the above)
