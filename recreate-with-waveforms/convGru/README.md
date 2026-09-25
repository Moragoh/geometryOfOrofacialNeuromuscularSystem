# Raw waveform vs. SPD matrices — Subject 1

**Question:** for classifying which phoneme/word was said, is it better to feed the network SPD covariance matrices or the raw EMG?

**Answer (Subject 1):** no evidence that raw data is better.
- **Same epoch budget (150):** the SPD-RNN is ahead in all 4 configs. The gap is outside noise only for voiced phonemes.
- **500 epochs for the raw model:** it catches up to within noise (−0.05 to +0.03).

So the SPD representation reaches the same accuracy with a third of the training. The raw model overfits: 100% train accuracy, with test accuracy flat after ~200 epochs.

## Results

Test accuracy, mean ± std over 10 seeds. SPD-RNN numbers are the paper's (gowda.pdf, Tables 10/11) and were not rerun.

| Task / manner | Chance | SPD-RNN (paper, 150 ep) | Conv+GRU, best test acc, epochs 1–150 | Conv+GRU, best test acc, epochs 1–500 | Conv+GRU, final epoch (500) | Best epoch, median (range) |
|---|---|---|---|---|---|---|
| Phonemes, voiced | 0.026 | **0.651 ± 0.030** | 0.545 ± 0.033 | 0.598 ± 0.043 | 0.547 ± 0.051 | 311 (184–495) |
| Phonemes, unvoiced | 0.026 | 0.574 ± 0.014 | 0.545 ± 0.027 | 0.589 ± 0.020 | 0.525 ± 0.042 | 374 (317–472) |
| Words, voiced | 0.028 | 0.788 ± 0.029 | 0.739 ± 0.043 | 0.816 ± 0.029 | 0.765 ± 0.041 | 410 (315–496) |
| Words, unvoiced | 0.028 | 0.669 ± 0.055 | 0.634 ± 0.039 | 0.678 ± 0.036 | 0.626 ± 0.036 | 343 (117–480) |

- **"Best test acc" columns:** the protocol the repo uses — keep the best test accuracy seen across epochs. It selects on the test set, so treat it as optimistic.
- **Epochs 1–150 column:** the matched-budget comparison with the paper. Training is deterministic per seed, so this equals a separate 150-epoch run; I checked seed 0 against an earlier 150-epoch run and it matched exactly.
- **Final epoch column:** the honest number. The paper doesn't report one.

**Reading the gaps:**
- **Noise level:** with 144–152 test samples, one accuracy has a binomial SE of ≈ ±0.04, and seed std reaches 0.055. Treat differences under ~0.08 as noise.
- **Seed-only ±:** the ± covers only the seed (weight init and shuffle order). The train/test split is fixed, so it understates the true uncertainty.
- **Best epochs are late and scattered:** the median best epoch is 311–410, and the ranges are wide. The test curves are flat by then, so the extra points from 500 epochs mostly come from picking a lucky epoch.

## What I did

Implemented `_notes/plan-raw-waveform-baseline.md`, with one change: 500 epochs instead of 150, because at 150 the raw model was still underfit (train accuracy ~0.8).

- **Same as the SPD-RNN notebooks (`allPhonemesRnn.ipynb`, `wordsRnn.ipynb`):**
  - Subject 1, the data files, per-trial per-channel z-score, labels
  - The 6/4 split: trials `0:3`+`5:8` train, `3:5`+`8:10` test
  - `BaseDataset`, batch size 32, `CrossEntropyLoss`
  - `spdLearning/trainTest.py` train/test functions
  - Adam (lr 1e-3, weight decay 1e-3)
- **Removed:** the covariance loop. The z-scored `(22, 7500)` trial goes straight into the model.
- **Model — `convGru.py`, 45k params (the SPD-RNN has ~147k):**
  1. Spatial conv (22→40)
  2. Depthwise temporal conv (kernel 25)
  3. Square, then `AvgPool1d(750, 150)`, which gives the same 46 windows as the SPD-RNN
  4. log
  5. Bidirectional GRU (hidden 64); each direction is averaged over time and the two are summed, as in `spdRnn.py`
  6. Dropout 0.5, then linear
- **Hyperparameters:** fixed in advance, not tuned.
- **Representation difference:** each window gives 40 learned-band powers wᵢᵀCwᵢ with no cross terms wᵢᵀCwⱼ, whereas the SPD-RNN sees the full projected covariance.
- **Runs:** 4 configs × 10 seeds (0–9) × 500 epochs on an RTX 3090, about 55 s per run, with deterministic cuDNN.

## Files

| File | |
|---|---|
| `convGru.py` | The model |
| `trainConvGru.ipynb` | Trains all 40 runs. Prints train/test accuracy every 10 epochs, the best epoch per seed, and plots accuracy over epochs. Writes `results.json`. |
| `resultsConvGru.ipynb` | Loads `results.json`: comparison table, bar chart, accuracy curves, caveats. |
| `results.json` | Per-seed, per-epoch train/test accuracy. No model weights are saved. |

## Rerunning

Data isn't committed. From the repo root, set up the environment and symlink Subject 1's files; the path below is this machine's copy:

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy torch tqdm jupyterlab matplotlib
cd recreate-with-waveforms
D=/media/oeste/research/WORKINGDATA/emg-data/Experiment1
mkdir -p Experiment1/Phoneme Experiment1/Words
for m in Voiced Unvoiced; do
  ln -s "$D/Phonemes/Subject1-3/${m}Subject1.npy" Experiment1/Phoneme/
  ln -s "$D/Words/Subject1-3/${m}Subject1.npy"    Experiment1/Words/
done
```

Then run `trainConvGru.ipynb` (~40 min on a GPU), followed by `resultsConvGru.ipynb`. The results notebook only needs `results.json`, not the data. Training writes progress to `progress.log` (not committed) for `tail -f`.
