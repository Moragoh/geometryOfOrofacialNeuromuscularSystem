# ResNet1D on raw EMG waveforms: high level plan

Parameters follow Wang et al. 2017, "Time series classification from scratch with deep neural networks" (arXiv 1611.06455). The paper does not state epochs, batch size, or the learning-rate schedule; those come from the authors' released `ResNet.py` (github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline).

## Layout (all new, nothing copied)

```
recreate-with-waveforms/
├── Experiment1/          # shared data links
└── resnet1d/
    ├── loadData.py       # load, z-score, split
    ├── resnet1d.py       # model
    ├── train.ipynb       # 4 configs × numberSeeds → results.json + progress.log
    └── results.py        # table vs the paper's SPD numbers
```

`train.ipynb` runs from inside `resnet1d/` and reads data from `../Experiment1/`.

## Step 1: Data (`loadData.py`) (VERIFED)

- Load `Experiment1/{Phoneme,Words}/{Voiced,Unvoiced}Subject1.npy`, shaped `(380 or 360, 22, 7500)`.
- Z-score each trial and each channel over time: `(x − mean) / std` along the time axis, as the gowda paper does (Look at existing scripts in repo to copy exactly how the original paper's repo does it).
- Labels: trial `i` belongs to class `i // 10` (classes are stored in blocks of 10).
- Split per class: repetitions 0–2 and 5–7 train, 3–4 and 8–9 test.
- Output: phonemes train `(228, 22, 7500)` / test `(152, 22, 7500)`; words train `(216, …)` / test `(144, …)`.

## Step 2: Model (`resnet1d.py`), from the paper

- **Residual block (paper's Eq. 2–3 and Fig. 1c, not the authors' code):** conv (kernel 8) → BN → ReLU → conv (5) → BN → ReLU → conv (3) → BN → ReLU, plus an identity shortcut, then ReLU after the addition. Stride 1 and `"same"` padding, so the length stays 7500.
  - The paper doesn't say what the shortcut does when the channel count changes (22→64, 64→128), where `h3 + x` is impossible. We use a 1×1 conv there (the projection shortcut of He et al. 2016, which the paper cites), no BN.
  - The authors' code differs: no ReLU after the third conv, BN on every shortcut, an extra BN on the network input, and (because of a channel-count check against the network input) a 1×1-conv shortcut in block 3 too.
- **Network:** block(22→64) → block(64→128) → block(128→128) → global average pooling over time → linear to 38 or 36 classes.
- No dropout, no weight decay (the paper uses neither for ResNet).

## Step 3: Training (`train.ipynb`), from the paper and its code

| Setting         | Value                                                                                  | Source |
| --------------- | -------------------------------------------------------------------------------------- | ------ |
| Optimizer       | Adam, lr 0.001, β₁ 0.9, β₂ 0.999, ε 1e-8                                               | paper  |
| Loss            | cross-entropy                                                                          | paper  |
| Epochs          | 500 for now (code uses 1500); final count decided from the train/test curves          | ours   |
| Batch size      | 32, same as the SPD notebooks (code uses `min(N_train / 10, 16)` → 16)                 | ours   |
| LR schedule     | halve the learning rate when training loss hasn't improved for 50 epochs, minimum 1e-4 | code   |
| Reported result | test accuracy at the epoch with the lowest training loss                               | paper  |

- Config cell near the top: `numberSeeds = 1` to start, `10` for the full run; `numberEpochs = 500`.
- Log three test accuracies per run:
  - Wang's rule: at the epoch with the lowest training loss (never looks at test).
  - The SPD repo's rule: best test accuracy across epochs (selects on test, optimistic).
  - Final epoch.
- Progress goes to `progress.log` while running; everything to `results.json` at the end.

## Step 4: Results (`results.py`)

A table per config: ResNet1D (mean ± std over seeds, all three numbers) vs the paper's SPDNet (Tables 8/9) and SPD-RNN (Tables 10/11) for Subject 1.

## Deviations from "unchanged"

- **Normalization:** Wang et al. normalize with training-set statistics. We use per-trial, per-channel z-scoring instead, to match the SPD pipeline, so the only difference from SPDNet/SPD-RNN is the representation and architecture. Train-set normalization is deferred.
- **Run time is an estimate:** about 3.8 billion multiply-adds per trial on the forward pass; roughly 0.5–1 s per epoch on the RTX 3090, so 4–8 min per run at 500 epochs (about 3–6 h for 4 configs × 10 seeds; 3× that at 1500). Not measured.
