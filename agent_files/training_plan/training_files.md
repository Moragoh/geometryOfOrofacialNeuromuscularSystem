# Files for training SPDNet and SPD-RNN (Experiment 1)

What to read to follow the path from the shared data (already re-referenced and Butterworth-filtered) to a trained model. Files are listed in reading order.

## Data (from OSF, not in the repo)
- `Experiment1/Phoneme/{Voiced,Unvoiced}Subject<N>.npy`: `(380, 22, 7500)`
- `Experiment1/Words/{Voiced,Unvoiced}Subject<N>.npy`: `(360, 22, 7500)`

## SPDNet (`spdLearning/`)

1. **A notebook (the entry point).** All four follow the same template:
   - `allPhonemesSPDNet.ipynb`: 38 phonemes
   - `consonantPhonemeSPDNet.ipynb`: 23 consonants (trials 0–229)
   - `vowelPhonemeSPDNet.ipynb`: 15 vowels (trials 230–379)
   - `wordsSPDNet.ipynb`: 36 words

   Read in order: the parameter cells (`subjectNumber`, `articulationManner`) → load the `.npy` and z-score each channel → one covariance matrix per trial, `1/7500 · X Xᵀ` → train/test split (repetitions 0–2 and 5–7 train, 3–4 and 8–9 test) → `BaseDataset` / `DataLoader` → model, `MixOptimizer` and `CrossEntropyLoss` → epoch loop.
2. **`spdLearning/spdNet.py`: the model.** `learnSPDMatrices` wraps `manifoldNet`: BiMap(22→22) → ReEig → BiMap(22→20) → ReEig → BiMap(20→16) → ReEig → LogEig → flatten to 256 → `nn.Linear(256, classes)`.
3. **`spdLearning/spdNN.py`: layer modules.** `BiMap`, `ReEig` and `LogEig`, thin wrappers around the functions in `functional.py`.
4. **`spdLearning/functional.py`: the math.**
   - `StiefelParameter`, `initBimapParameter`
   - `bimap` (computes `Wᵀ X W`)
   - `modeigForward` / `modeigBackward` (eigenvalue functions and their hand-written gradients)
   - `ReOp` (floor at 1e-3), `LogOp`
5. **`spdLearning/optimizers.py`: the optimizer.** `MixOptimizer` sends `StiefelParameter`s to `StiefelOptim`: project the gradient onto the tangent space, take a step, re-orthonormalize with `gramSchmidt`. Every other parameter goes to `torch.optim.SGD`.
6. **`spdLearning/trainTest.py`: epoch loops.** `trainOperation` / `testOperation` return the loss and accuracy for one epoch.

## SPD-RNN (`manifoldRnn/`)

1. **A notebook (the entry point).** All four follow the same template:
   - `allPhonemesRnn.ipynb`: 38 phonemes
   - `consonantPhonemesRnn.ipynb`: 23 consonants
   - `vowelPhonemeRnn.ipynb`: 15 vowels
   - `wordsRnn.ipynb`: 36 words

   Same flow as SPDNet, except each trial becomes **46 covariance matrices** from 750-sample windows every 150 samples, giving `(46, 22, 22)`. There are two optimizers: `StiefelOptim` for `model.CNN` and `Adam` for `model.RNN`.
2. **`manifoldRnn/spdRnn.py`: the model** (partly copied from Jeong et al. 2023).
   - `spdRnnNet` = `spdNet` (named `CNN`) + `rnnNet` (named `RNN`).
   - `spdNet`: BiMap(22→22) → BiMap(22→20), applied to each of the 46 matrices. No ReEig or LogEig.
   - `rnnNet.cholDe`: Cholesky-decomposes each 20×20 matrix into a diagonal part (20 values) and a strictly-lower-triangle part (190 values).
   - `RGRUCell`: a GRU cell. The `diag=True` version works in log space with `PosLinear` (absolute-valued weights) to keep the diagonal positive. The `diag=False` version is a standard GRU.
   - `ODEFunc` / `odefunc`: a small network whose output `torchdiffeq.odeint` integrates (Euler method) to evolve the hidden state between steps.
   - `rnnNet.forward`: runs forward and backward passes over the 46 steps, averages each pass's hidden states over time, adds the two, then applies the `cls` linear layer.
3. **`manifoldRnn/spdNN.py`, `manifoldRnn/functional.py`: the `BiMap` layer.** Same content as the `spdLearning/` versions.
4. **`manifoldRnn/optimizers.py`: `StiefelOptim`.** Same content as the `spdLearning/` version. The notebook uses `StiefelOptim` directly, not `MixOptimizer`.
5. **`manifoldRnn/trainTest.py`: epoch loops.** Same as `spdLearning/trainTest.py`, except `trainOperation` takes two optimizers and steps both.

## External libraries used by the model code
`torch` (both models), `torchdiffeq` (SPD-RNN's ODE step only), `numpy` (notebooks).
