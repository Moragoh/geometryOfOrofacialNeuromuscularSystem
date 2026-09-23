# Trial labeling in Experiment 1 (words and phonemes)

There are no label files for Experiment 1. The class of each trial comes only from where it sits in the array.

## The rule
Files: `Experiment1/Words/{Voiced,Unvoiced}Subject<N>.npy` → `(360, 22, 7500)` (36 words × 10 reps)
       `Experiment1/Phoneme/{Voiced,Unvoiced}Subject<N>.npy` → `(380, 22, 7500)` (38 phonemes × 10 reps)

Trials are stored grouped by class, 10 consecutive repetitions per class:

```python
class_index = trial_index // 10       # which word / phoneme
repetition_index = trial_index % 10   # which of its 10 repetitions (0–9)
```

- Trials 0–9 are class 0, trials 10–19 are class 1, and so on.
- Example: `DATA[57]` in a words file is class 5 ("matted"), repetition 7.
- The notebooks build labels the same way: `np.array([[i] * 10 for i in range(numberClasses)]).reshape(-1)`.
- The notebooks' train/test split works per class: repetitions 0–2 and 5–7 train, 3–4 and 8–9 test.

## Assumptions (not verified from the data itself)
- Trials really are stored grouped and in class order. The code assumes it, but the files carry no labels to check against.
- Every subject's file uses the same class order. The code applies the same mapping to every subject but never pools them. If you combine subjects, you are relying on this.
- Voiced and Unvoiced files are separate problems with the same class order.

## Words (36 classes)
Source: docstring in `wordsSPDNet.ipynb`; matches the paper's §2.1.3 order.

| # | Word | # | Word | # | Word | # | Word |
|---|---|---|---|---|---|---|---|
| 0 | eager | 9 | rook | 18 | this | 27 | jeep |
| 1 | lift | 10 | folder | 19 | tango | 28 | ship |
| 2 | eight | 11 | block | 20 | doubt | 29 | beige |
| 3 | edge | 12 | fun | 21 | not | 30 | yes |
| 4 | cap | 13 | mop | 22 | pretty | 31 | echo |
| 5 | matted | 14 | pod | 23 | xerox | 32 | gold |
| 6 | tub | 15 | very | 24 | rodent | 33 | sing |
| 7 | box | 16 | went | 25 | limb | 34 | uh-oh |
| 8 | rune | 17 | throat | 26 | batch | 35 | hiccup |

## Phonemes (38 classes)
Source: docstring in `allPhonemesSPDNet.ipynb` (consonants) and the paper's §2.1.2 (vowels).

Consonants 0–22 (trials 0–229):

| # | Phoneme | # | Phoneme | # | Phoneme |
|---|---|---|---|---|---|
| 0 | Baa | 8 | Daa | 16 | Kaa |
| 1 | Paa | 9 | Naa | 17 | Gaa |
| 2 | Maa | 10 | Saa | 18 | NGaa |
| 3 | Faa | 11 | Zaa | 19 | Yaa |
| 4 | Vaa | 12 | Chaa | 20 | Raa |
| 5 | Thaa | 13 | Shaa | 21 | Laa |
| 6 | Dhaa | 14 | Jhaa | 22 | Waa |
| 7 | Taa | 15 | Zhaa | | |

Vowels 23–37 (trials 230–379; `vowelPhonemeSPDNet.ipynb` reads from trial 230):

| # | Vowel | # | Vowel | # | Vowel |
|---|---|---|---|---|---|
| 23 | OY (bOY) | 28 | AY (mY) | 33 | IH (It) |
| 24 | OW (nOW) | 29 | AE (At) | 34 | AH (HUt) |
| 25 | AO (OUght) | 30 | EH (mEt) | 35 | UW (fOOD) |
| 26 | AA (fAther) | 31 | EY (mAte) | 36 | ER (hER) |
| 27 | AW (cOW) | 32 | IY (mEET) | 37 | UH (hOOD) |

**Caveat on vowels:** the notebook docstring lists only 13 vowels (23–35) and leaves out AW and AY, even though the data has 15. The table above follows the paper's order, which includes them. That order is likely correct, but nothing in the repo confirms it. If the recordings follow the notebook's list instead, every vowel after AA is shifted.
