# EEG–Audio Topology in Infant Speech Processing

Topological Data Analysis of infant EEG during speech. We test whether persistent homology captures condition dependent structure and whether EEG topology is coupled to audio topology.

### Data

- 45 infants (3-5 months), 1,416 recordings (710 slow audio + 706 fast audio)
- for every audio the recording included 47-channel EEG at 250 Hz, audio at 44,100 Hz 
- MATLAB files: `bbXX_utYY.mat` with `subeeg`, `y`, `Fs`


### Pipeline

1) Preprocess EEG and audio: band-pass filter EEG (47 channels) into 5 frequency bands, segment into 1 s windows; resample audio, extract amplitude envelope, filter into the same bands.
   
**Parte 1.**

2) Build EEG connectivity graphs: pairwise Pearson correlation between channels: create matrix with correlation (or distance) per window per band.
3) Compute persistent homology: Rips filtration on EEG distance matrices and Takens embeddings of audio windows. from persistence diagrams (H0, H1) we extract different scalar features (persistence statistics, entropy). [revisar pendiente]
4) Classify slow vs. fast EEG: Random Forest on TDA features with group cross-validation, permutation test (n = 1000), and subject-level bootstrap 95% CI.

**Parte 2.**

5) Compare EEG–audio topology: Wasserstein distance between EEG and audio persistence diagrams per band; within-subject Wilcoxon signed-rank test (slow vs. fast), FDR-corrected.
6) Validate coupling specificity: compare Wasserstein distances between matched pairs (EEG ↔ same-condition audio) against mismatched pairs (EEG ↔ opposite-condition audio) within each subject to confirm condition-specific coupling.


### Project Structure

```
tda-eeg-audio/
├── data/                   # Raw data (slow/ and fast/)
├── preprocessed/           # Band-filtered windows
├── graphs/                 # Connectivity distance matrices per recording
├── features/               # Cached TDA features (X.npy, y.npy, subjects.npy)
├── results/       
├── scripts/
│   ├── utils.py                      
│   ├── tda_eeg_audio_comparison.py   # Wasserstein EEG-audio comparison
│   ├── matched_vs_mismatched.py      # Matched/mismatched control
│   ├── classification_rerun.py       # Classification (fast vs slow) tda
│   └── tda_eeg_classification_v2.py  
├── notebooks/              # Pipeline notebooks (EDA, preprocessing, graph construction)         
├── requirements.txt
└── README.md
