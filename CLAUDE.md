# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper [Geometry of orofacial neuromuscular signals: speech articulation decoding using surface electromyography](https://arxiv.org/pdf/2411.02591). Surface EMG from 22 facial channels is turned into **SPD (symmetric positive definite) covariance matrices**, and classification happens with Riemannian geometry on the SPD manifold rather than in Euclidean space. Everything else follows from that choice.

## Environment

No `requirements.txt` / `environment.yml`. Notebooks were authored on Python 3.10.14. Imports across the repo require:

`numpy`, `scipy`, `torch`, `torchdiffeq`, `scikit-learn`, `scikit-learn-extra` (`sklearn_extra.cluster.KMedoids`), `matplotlib`, `tqdm`

`scikit-learn-extra` is the fragile one — it pins older `scikit-learn`/numpy versions.

## Data (not in the repo)

Download the ~40 GB dataset from [OSF](https://osf.io/ym5jd/) and unzip so that `Experiment1/` and `Experiment2/` sit at the repo root — notebooks use **relative paths from the repo root**, and both directories are gitignored.

```
Experiment1/Phoneme/{Voiced,Unvoiced}Subject<N>.npy   # (380, 22, 7500) = 38 phonemes x 10 trials
Experiment1/Words/{Voiced,Unvoiced}Subject<N>.npy     # (360, 22, 7500) = 36 words x 10 trials
Experiment1/orofacialMovements/Subject<N>.npy
Experiment2/Subject<N>/trainSet.npy                   # NATO alphabet, 26 classes
Experiment2/Subject<N>/{rainbow,grandfather}Passage.npy + ...Labels.npy
Experiment2/Subject<N>/grandfatherPassageContinuousSentences.npy
Experiment2/Subject<N>/{spdNet.pt,rnn.pt}             # checkpoints written by the Part2 *Train notebooks
```

Arrays are `(trials, 22 channels, 7500 samples)`, already bandpass-filtered (3rd-order Butterworth, 80–1000 Hz). "Voiced" = audible articulation, "Unvoiced" = silent. Subject 11 is absent from the subject lists in the MDM notebooks. Passage labels ship separately per subject because a software glitch dropped characters for some subjects.

## Running

There is no CLI, no test suite, and no build step — every experiment is a notebook at the repo root that imports the three local packages. Run Jupyter **from the repo root** so both the relative data paths and the package imports resolve:

```bash
jupyter lab
```

Notebooks are parameterized by editing cells near the top, not by CLI args: `subjectNumber`, `articulationManner` (`"Voiced"` / `"Unvoiced"`), `numberEpochs`, and `dev`. A full SPDNet run is `numberEpochs = 1000`; the RNN runs 150.

**`dev` matters.** `spdLearning` (SPDNet) notebooks set `dev = "cpu"`; `manifoldRnn` notebooks set `dev = "cuda:0"` and `manifoldRnn/spdRnn.py` also **hardcodes `device = 'cuda:0'`** in `rnnNet.__init__`, so the RNN path needs a CUDA GPU unless you change both.

## Architecture

### Shared pipeline (repeated inline in every notebook, not factored into a module)

1. Load `.npy` → per-channel z-score over the time axis.
2. Build covariance matrices `1/T * X @ X.T`:
   - **SPDNet / MDM**: one `22x22` matrix per whole 7500-sample trial.
   - **RNN**: a *sequence* of 46 matrices per trial via a sliding window (750 samples, 150-sample hop), giving `(trials, 46, 22, 22)`.
3. Fixed train/test split per class — trials `0:3` + `5:8` train, `3:5` + `8:10` test (6 train / 4 test per class).
4. Wrap in the notebook-local `BaseDataset` + `DataLoader`, train, track best test accuracy.

### `spdLearning/` — SPDNet (static covariance → class)

- `spdNet.py`: `learnSPDMatrices` → `manifoldNet` = three `BiMap` layers (22→22→20→16) each followed by `ReEig`, then `LogEig` flattens the 16x16 output to 256 dims into a `nn.Linear`. Change `numberChannels` and you must change `BiMap(1, 22, 22)` here too.
- `spdNN.py` / `functional.py`: the SPD layer primitives — `BiMap` (bilinear `Wᵀ X W`, weights are `StiefelParameter`), `ReEig` (eigenvalue rectification), `LogEig` (matrix log to the tangent space), with hand-written forward/backward via `modeigForward`/`modeigBackward`. Copied from Brooks et al. 2019 / Huang & Van Gool 2017.
- `optimizers.py`: `MixOptimizer` splits parameters — `StiefelParameter`s get Riemannian SGD on the Stiefel manifold (tangent projection + retraction by Gram–Schmidt), everything else gets a standard torch optimizer. **Plain `torch.optim` alone will break the manifold constraint.**
- `trainTest.py`: `trainOperation` / `testOperation`, single optimizer signature.

### `manifoldRnn/` — SPD-RNN (covariance sequence → class)

`spdNN.py`, `functional.py`, `optimizers.py` are **byte-identical copies** of the `spdLearning` versions (one blank line apart). Fix a bug in one and it must be fixed in the other. What differs:

- `spdRnn.py`: `spdRnnNet` = `spdNet` (two `BiMap` layers 22→22→20, no `ReEig`/`LogEig`) feeding `rnnNet`. `rnnNet` Cholesky-decomposes each SPD matrix into diagonal and strictly-lower-triangular parts and runs *separate* recurrences over each — `RGRUCell(diag=True)` operates multiplicatively in log-space with `PosLinear` (abs-valued weights) to keep the diagonal positive; `RGRUCell(diag=False)` is a conventional GRU for the lower triangle. A neural ODE (`torchdiffeq.odeint`, Euler) evolves the hidden state between timesteps. Bidirectional: forward and reversed passes are mean-pooled and summed. Adapted from Jeong et al. 2023.
- `trainTest.py`: same API as `spdLearning`'s **except** `trainOperation` takes **two** optimizers — `StiefelOptim` for `model.CNN` and `Adam` for `model.RNN`. Not interchangeable with the `spdLearning` version.

### `basicOperations/manifoldOperations.py` — geometry without learning

Implements Lin (2019) Cholesky-based SPD geometry in NumPy: `matrixDistance` (log-Cholesky geodesic distance), `frechetMean` (manifold mean), `tSNEmbedding` (t-SNE on a precomputed geodesic distance matrix), `unsupervised.kMedoids` (PAM clustering on geodesic distances). Used by the MDM, t-SNE, and unsupervised notebooks — no PyTorch involved.

## Notebook map

| Pattern | Model | Package |
|---|---|---|
| `*SPDNet.ipynb`, `words/vowel/consonant/allPhonemes` | SPDNet, CPU | `spdLearning` |
| `*Rnn.ipynb` | SPD-RNN, CUDA | `manifoldRnn` |
| `*MDM.ipynb` | Minimum-distance-to-mean, loops all subjects, no training | `basicOperations` |
| `orofacialTsne.ipynb`, `unsupervisedOrofacialClassification.ipynb` | t-SNE / k-medoids on the manifold | `basicOperations` |
| `Part2_NATOAlphabets*Train.ipynb` | Trains on `trainSet.npy` and **saves** `spdNet.pt` / `rnn.pt` | both |
| `Part2_NATOAlphabets_{Rainbow,Grandfather}*.ipynb` | **Loads** those checkpoints, evaluates top-1..top-5 on passage data | both |
| `Part2ContinuousSentences*.ipynb` | Continuous sentence decoding | both |

The Part2 passage notebooks are evaluation-only — they require the matching `*Train` notebook to have run for that subject first, and the checkpoint's class count (26) must match the model constructed in the eval notebook.
