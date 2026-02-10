# EEG–Audio Topology in Infant Speech Processing

Topological Data Analysis of infant EEG during speech. We test whether persistent homology captures condition-dependent structure and whether EEG topology is coupled to audio topology.

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
│   ├── utils.py                      
│   ├── tda_eeg_audio_comparison.py   # Analysis 2: Wasserstein EEG-audio comparison
│   ├── matched_vs_mismatched.py      # Analysis 3: Matched-vs-mismatched control
│   ├── classification_rerun.py       # Analysis 1: RF classification
│   └── tda_eeg_classification_v2.py  
├── notebooks/              # Pipeline notebooks (EDA, preprocessing, graph construction)
├── paper/
│   ├── paper.tex
│   └── figures/           
├── requirements.txt
└── README.md
```

## Pipeline

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
python scripts/matched_vs_mismatched.py        # -> results/matched_vs_mismatched.json
```

## Methods

1. **Preprocessing**: Butterworth bandpass (order 4) into 5 bands (delta, theta, alpha, beta, gamma). 1s windows, 75% overlap.
2. **Connectivity graphs**: Pearson correlation -> Euclidean distance `d = sqrt(2(1-r))`. 47x47 matrices per window.
3. **TDA**: Vietoris-Rips persistent homology (H0, H1) via Ripser. 11 scalar features per dimension per band.
4. **Audio TDA**: Hilbert envelope -> per-band filtering -> Takens embedding (dim=3) -> persistence diagrams.
5. **Classification**: Random Forest (100 trees, max_depth=10), StratifiedGroupKFold (k=5), permutation test (1000 iter).
6. **EEG-audio coupling**: 1-Wasserstein distance between EEG and audio persistence diagrams. Wilcoxon signed-rank, FDR-corrected.
7. **Matched-vs-mismatched control**: W(EEG, Audio_same) vs W(EEG, Audio_opposite) per subject per band.
