# Topological EEG--Audio Coupling in Infant Speech Processing

Topological Data Analysis (TDA) of infant EEG functional connectivity during speech processing. We test whether persistent homology captures condition-dependent topological signatures and whether brain connectivity topology is structurally coupled to auditory stimulus topology.

## Results

### Classification (Analysis 1)

| Metric | Value |
|--------|-------|
| Accuracy | 73.1% +/- 2.1% |
| F1 | 0.732 |
| ROC-AUC | 0.804 |
| p-value | < 0.001 (permutation test) |
| 95% CI | [69.8%, 75.6%] |
| Top band | Gamma (44.5%) |

### EEG--Audio Topological Coupling (Analysis 2)

Wasserstein distances between EEG and audio persistence diagrams (H1) are significantly smaller in the slow condition for delta and theta (p_FDR < 0.002).

### Matched-vs-Mismatched Control (Analysis 3)

| Band | Direction | p_FDR | Cohen's d | % matched < mismatch |
|------|-----------|-------|-----------|---------------------|
| Theta | matched < mismatch | 1.8e-9 | -0.88 | 93% |
| Alpha | matched < mismatch | 4.1e-7 | -0.88 | 80% |
| Beta | matched < mismatch | 2.5e-11 | -2.07 | 93% |
| Delta | matched > mismatch | 1.5e-8 | +1.09 | 11% |
| Gamma | n.s. | 0.54 | -0.07 | 33% |

Theta, alpha, and beta show genuine condition-specific coupling. Delta is reversed (reflects intrinsic audio differences). Gamma is not significant.

## Data

- 45 infants (3-5 months), 1,416 recordings (710 slow + 706 fast)
- 47-channel EEG at 250 Hz, audio at 44,100 Hz
- MATLAB files: `bbXX_utYY.mat` with `subeeg`, `y`, `Fs`

## Project Structure

```
tda-eeg-audio/
├── data/                   # Raw .mat files (slow/ and fast/)
├── preprocessed/           # Band-filtered windows
├── graphs/                 # Connectivity distance matrices per recording
├── features/               # Cached TDA features (X.npy, y.npy, subjects.npy)
├── results/                # JSON results + figures
├── scripts/
│   ├── tda_eeg_audio_comparison.py   # Analysis 2: Wasserstein EEG-audio comparison
│   ├── classification_rerun.py       # Analysis 1: RF classification
│   └── tda_eeg_classification_v2.py  # Feature extraction (legacy)
├── notebooks/              # Pipeline notebooks (EDA, preprocessing, graph construction)
├── paper/
│   ├── paper_revised.tex   # Current manuscript
│   ├── paper_revised.pdf   # Compiled PDF
│   ├── respuesta_comentarios.md
│   └── figures/            # Publication figures
├── requirements.txt
└── README.md
```

## Pipeline

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn ripser persim matplotlib seaborn statsmodels tqdm
```

### Full pipeline (from raw data)

```bash
# 1. Preprocessing and graph construction (notebooks)
#    notebooks/1_preprocesamiento.ipynb  -> preprocessed/
#    notebooks/2_graph_construction.ipynb -> graphs/

# 2. Feature extraction
python scripts/tda_eeg_classification_v2.py    # -> features/

# 3. EEG-audio Wasserstein comparison (~10 min)
python scripts/tda_eeg_audio_comparison.py     # -> results/eeg_audio_tda_comparison.json

# 4. Classification with permutation test (~15 min)
python scripts/classification_rerun.py         # -> results/results_summary.json

# 5. Matched-vs-mismatched control (~10 min)
python matched_vs_mismatched.py                # -> results/matched_vs_mismatched.json
```

### Compile paper

```bash
cd paper && pdflatex paper_revised.tex && pdflatex paper_revised.tex
```

## Methods

1. **Preprocessing**: Butterworth bandpass (order 4) into 5 bands (delta, theta, alpha, beta, gamma). 1s windows, 75% overlap.
2. **Connectivity graphs**: Pearson correlation -> Euclidean distance `d = sqrt(2(1-r))`. 47x47 matrices per window.
3. **TDA**: Vietoris-Rips persistent homology (H0, H1) via Ripser. 11 scalar features per dimension per band.
4. **Audio TDA**: Hilbert envelope -> per-band filtering -> Takens embedding (dim=3) -> persistence diagrams.
5. **Classification**: Random Forest (100 trees, max_depth=10), StratifiedGroupKFold (k=5), permutation test (1000 iter).
6. **EEG-audio coupling**: 1-Wasserstein distance between EEG and audio persistence diagrams. Wilcoxon signed-rank, FDR-corrected.
7. **Matched-vs-mismatched control**: W(EEG, Audio_same) vs W(EEG, Audio_opposite) per subject per band.

## Author

Maria Ignacia Gothe -- Pontificia Universidad Catolica de Chile (mgothe@uc.cl)

## References

- Edelsbrunner, H. and Harer, J.L. (2010). *Computational Topology*. AMS.
- Carlsson, G. (2009). Topology and data. *Bull. Amer. Math. Soc.*, 46(2):255-308.
- Giraud, A.L. and Poeppel, D. (2012). *Nat. Neurosci.*, 15(4):511-517.
