# Plan — raw-waveform Conv+GRU vs. SPD-RNN (Subject 1)

Status: **plan only**, nothing implemented. Comparison baseline = the paper's published SPD-RNN tables.

## 1. The question, stated precisely

Holding subject, manner, task, train/test split, z-scoring, **and the 46-window time grid** fixed,
does a network fed the **raw `(22, 7500)` waveform** match or beat the paper's **SPD-RNN**, which is
fed a sequence of 46 windowed `22×22` covariance matrices?

The SPD-RNN is the paper's strongest model, and it beats SPDNet in every Subject 1 row below —
evidence that temporal order carries class information. So the raw model is sequential too, and
the only variable is the per-window representation:

- **SPD-RNN:** fixed broadband (80–1000 Hz) `22×22` covariance per window.
- **Conv+GRU:** learned spatial filters + learned frequency band, log-power per window.

## 2. Baseline numbers to beat (Subject 1, SPD-RNN, gowda.pdf)

`B.2.3` = section 2.3.7 = **SPD-RNN** (Tables 10, 11). Values are "mean ± std over 10 random seeds".

| Task | Manner | SPD-RNN (T10/T11) |
|---|---|---|
| All phonemes (38-way, chance 0.026) | Audible | 0.651 ± 0.030 |
| All phonemes | Silent | 0.574 ± 0.014 |
| All words (36-way, chance 0.028) | Audible | 0.788 ± 0.029 |
| All words | Silent | 0.669 ± 0.055 |

**What counts as "beating" it.** 152 (phonemes) / 144 (words) test samples at ~65% accuracy give
a binomial standard error of ≈ ±0.04, and the paper's seed std is up to 0.055. Gaps smaller than
~0.08 are not evidence of a real difference. State this next to the results.

## 3. Data

**Location:** `/media/oeste/research/WORKINGDATA/emg-data/Experiment1` (already on disk, no
download needed).

**Its layout does not match the notebooks' paths.** The notebooks load
`Experiment1/Phoneme/<Manner>Subject1.npy` and `Experiment1/Words/<Manner>Subject1.npy`. On disk
the files are at:

```
.../Experiment1/Phonemes/Subject1-3/{Voiced,Unvoiced}Subject1.npy   # "Phonemes", plural, + subfolder
.../Experiment1/Words/Subject1-3/{Voiced,Unvoiced}Subject1.npy
```

So a single directory symlink won't work. Instead, create a real `Experiment1/` at the repo root
(already gitignored) and symlink the 4 files into it:

```bash
D=/media/oeste/research/WORKINGDATA/emg-data/Experiment1
mkdir -p Experiment1/Phoneme Experiment1/Words
for m in Voiced Unvoiced; do
  ln -s "$D/Phonemes/Subject1-3/${m}Subject1.npy" Experiment1/Phoneme/
  ln -s "$D/Words/Subject1-3/${m}Subject1.npy"    Experiment1/Words/
done
```

Every `np.load` line copied from the original notebooks then works unchanged.

## 4. What is reused verbatim

Source notebooks: `allPhonemesRnn.ipynb` and `wordsRnn.ipynb`.

| Reused | Where | Note |
|---|---|---|
| Load + per-channel z-score | cell 8 | `mean/std` over `axis=-1`, `+1e-5`. Per-trial-per-channel, so no leakage. Copy unchanged. |
| Label + `Indices` construction | cell 8 | copy unchanged |
| 6/4 split loops | cell 9 (phonemes) / 9 (words) | train `[:3]` + `[5:8]`, test `[3:5]` + `[8:10]`. Only the feature shape changes: `(N, 46, 22, 22)` → `(N, 22, 7500)`. |
| `BaseDataset` | cell 4 | copy as-is (`.astype('float32')`) |
| `DataLoader` config | cell 9/10 | `batch_size=32`, `shuffle=True` train / `False` test |
| **`spdLearning/trainTest.py`** | `trainOperation` / `testOperation` | **Import this one, not `manifoldRnn`'s.** It takes a single optimizer and only calls `model(data)`, so it works with plain `Adam`. `manifoldRnn/trainTest.py` needs two optimizers (Stiefel + Adam). |
| Epoch loop + `maxValue` tracking | cell 11/12 | copy the protocol exactly (see §6) |

**The one edit:** delete the sliding-window covariance loop (`slicedMatrices[j, i] = 1/750 * ...`)
and put the raw z-scored trial straight into the split. The windowing moves inside the model
(§5), using the **same** windows.

## 5. Architecture — Conv+GRU

### Design

```
input             (B, 22, 7500)
spatial conv      Conv1d(22 → 40, kernel=1, bias=False)                        # 40 learned spatial filters
temporal conv     Conv1d(40 → 40, kernel=25, groups=40, padding=12, bias=False) # learned band per filter (5 ms taps @ 5 kHz)
square            x ** 2                                                        # phase → power
window pool       AvgPool1d(kernel=750, stride=150)                             # → (B, 40, 46)
log               log(clamp(x, min=1e-6))
transpose         → (B, 46, 40)
GRU               nn.GRU(40 → 64, 1 layer, bidirectional=True, batch_first=True)
aggregate         mean over the 46 steps per direction, then sum the two directions → (B, 64)
dropout           p=0.5
linear            Linear(64 → 38 or 36)
```

### Why each piece

- **Spatial → temporal → square → pool** is the ShallowConvNet block. It computes the power of
  `wᵢᵀX` in a learned frequency band for each window, i.e. `wᵢᵀ C_window wᵢ`. That's a learned
  projection of the same windowed covariance the SPD-RNN gets, with the band learned instead of
  fixed. Spatial filtering comes first because it's far cheaper (880 params vs. ~35k for the
  textbook temporal-first order); both steps are linear, so their order doesn't change what
  the model can represent.
- **`AvgPool1d(750, 150)` reproduces the SPD-RNN's windows exactly.** The notebook windows are
  `[i·150, i·150 + 750)` for `i = 0..45`. On a length-7500 input this pool gives the same 46
  windows. `padding=12` on the temporal conv keeps the length at 7500; without it the length
  drops to 7476 and you get 45 windows.
- **Bidirectional, mean-pooled, summed** mirrors `rnnNet.forward` in `manifoldRnn/spdRnn.py`
  (forward and reversed passes each averaged over time, then added). So the aggregation isn't a
  difference between the models either.
- **Known representational difference:** each window gives 40 powers `wᵢᵀCwᵢ` and no cross terms
  `wᵢᵀCwⱼ`, whereas the SPD-RNN sees the whole projected 20×20 matrix via Cholesky. One sentence
  in the write-up.

### Parameter counts (hand-computed; confirm with the `numParams` print)

| Model | Phonemes (38) | Words (36) |
|---|---|---|
| SPD-RNN (`spdRnnNet`) | ≈146.8 k | ≈146.6 k |
| Conv+GRU (this plan) | ≈45.1 k | ≈44.9 k |

Conv+GRU breakdown: spatial 880, temporal 1 000, bidirectional GRU 40 704, linear 2 470 / 2 340.
SPD-RNN: BiMaps 924, two directions of (diagonal RGRU 2 460 + lower-triangular RGRU 48 006 +
ODE net 20 874) = 142 680, classifier 3 192 / 3 024.

The raw model is about 3× smaller. That's deliberate: if it wins, the win can't be put down to
capacity. If it loses, capacity is a possible reason; report that openly rather than tuning to
close the gap.

### Fixed up front

These hyperparameters (40 filters, kernel 25, hidden 64, dropout 0.5) are chosen **before**
seeing any test numbers and are not tuned. With test-set selection (§6), trying variants would
inflate the result.

## 6. Protocol — matching the SPD-RNN notebooks, including their flaws

**The repo selects on the test set.** The epoch loop keeps `maxValue = max over epochs of test
accuracy`. There is no validation set. Replicate it, because comparing an honest protocol with
the paper's peeked one would bias against the raw model.

- **Primary metric (paper-comparable):** max over epochs of test accuracy, per seed.
- **Secondary metric (honest):** final-epoch test accuracy, per seed. Report alongside.

**Epoch budget: 150**, the same as `allPhonemesRnn.ipynb` / `wordsRnn.ipynb`. Under max-over-epochs,
more epochs can only raise the score, so the budget must match.

**Optimizer:** one `Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)`. These are the SPD-RNN
notebooks' Adam settings for its RNN part. The SPD-RNN's `StiefelOptim(lr=0.05)` exists only to
keep BiMap weights on the Stiefel manifold, so it has no counterpart here. `CrossEntropyLoss`,
`batch_size=32`.

**Seeds:** 0–9. Before building each model, call `torch.manual_seed(s)`, `np.random.seed(s)`,
`torch.cuda.manual_seed_all(s)`. Set `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False`; otherwise the cuDNN GRU isn't reproducible run to run.
Report mean ± std to 3 decimals (`0.xxx ± 0.xxx`).

The split is deterministic, so a seed only changes weight initialization and shuffle order, the
same variance source the paper's ± reflects. That ± understates the true uncertainty (held-out
trials are never resampled); one sentence in the write-up.

**Device:** `cuda:0` (RTX 3090 present). Unlike `spdRnn.py`, nothing in the new model hardcodes
the device.

## 7. Run matrix

4 configs × 10 seeds = **40 runs** × 150 epochs. The model is small and the data is ~200 MB per
file in float32, so each run should take minutes on the 3090.

| # | Subject | Manner | Task | Classes | Data file (via §3 symlinks) |
|---|---|---|---|---|---|
| 1 | 1 | Voiced | all phonemes | 38 | `Experiment1/Phoneme/VoicedSubject1.npy` |
| 2 | 1 | Unvoiced | all phonemes | 38 | `Experiment1/Phoneme/UnvoicedSubject1.npy` |
| 3 | 1 | Voiced | all words | 36 | `Experiment1/Words/VoicedSubject1.npy` |
| 4 | 1 | Unvoiced | all words | 36 | `Experiment1/Words/UnvoicedSubject1.npy` |

## 8. File layout

The notebooks stay at the repo root so the data paths and package imports resolve, matching repo
convention. The model goes in a new package: the directory name must be a valid Python identifier,
so not `recreate-with-waveforms`.

```
rawWaveform/
    __init__.py
    convGru.py            # convGruNet(classifications); mirrors manifoldRnn/spdRnn.py in shape
allPhonemesConvGru.ipynb  # mirrors allPhonemesRnn.ipynb cell-for-cell
wordsConvGru.ipynb        # mirrors wordsRnn.ipynb cell-for-cell
```

Each notebook contains the copied §4 preprocessing (minus the covariance loop), a
`for seed in range(10):` loop around model construction and the epoch loop, recording primary and
secondary metrics per seed, and a final `mean ± std` print. It's parameterized by editing top cells
(`subjectNumber`, `articulationManner`), matching repo convention.

## 9. Setup before running

1. **Data symlinks:** run the §3 commands.
2. **Python env:** no `venv` exists yet. Create `venv`, activate it, install
   `numpy torch tqdm jupyterlab` (`scipy`, `scikit-learn`, `torchdiffeq`, and `scikit-learn-extra`
   aren't needed for this plan).
3. Launch `jupyter lab` from the repo root.

## 10. Deliverable

| Task / manner | SPD-RNN (paper) | Conv+GRU, max-over-epochs | Conv+GRU, final epoch |
|---|---|---|---|

Include parameter counts for both models, the ±0.04 significance note from §2, the no-cross-terms
note from §5, and the fixed-split caveat from §6.
