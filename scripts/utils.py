"""
Shared utility functions for TDA EEG-Audio pipeline.
=====================================================
Signal processing, TDA computation, and Wasserstein distance helpers
used across multiple analysis scripts.
"""

import numpy as np
from scipy import signal as sig_proc
from scipy import io as sio
from ripser import ripser
from persim import wasserstein as wasserstein_distance
from pathlib import Path
import warnings


# ── Project paths ──
def get_project_root():
    """Return the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


# ── TDA parameters ──
MAX_DIM = 1
MAX_EDGE_LENGTH = 2.0
TAKENS_DIM = 3
TAKENS_SUBSAMPLE = 2

# ── Frequency bands ──
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}

# ── Sampling rates ──
FS_AUDIO = 44100
FS_EEG = 250


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def load_audio(mat_path):
    """Load audio signal from a .mat file."""
    mat = sio.loadmat(str(mat_path))
    y = mat['y']
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float64)


def compute_envelope(s, fs):
    """Compute the amplitude envelope of a signal via Hilbert transform."""
    analytic = sig_proc.hilbert(s)
    env = np.abs(analytic)
    nyq = fs / 2
    cutoff = min(50, nyq * 0.9)
    b, a = sig_proc.butter(4, cutoff / nyq, btype='low')
    return sig_proc.filtfilt(b, a, env)


def bandpass_filter(s, fs, low, high):
    """Apply a 4th-order Butterworth bandpass filter."""
    nyq = fs / 2
    lo = max(low / nyq, 0.001)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return s
    b, a = sig_proc.butter(4, [lo, hi], btype='band')
    return sig_proc.filtfilt(b, a, s)


def resample_audio(audio, fs_audio=FS_AUDIO, fs_target=FS_EEG):
    """Resample audio from fs_audio to fs_target using rational resampling."""
    return sig_proc.resample_poly(audio, fs_target, fs_audio)


def create_windows(s, win_samples, step_samples):
    """Create overlapping windows from a 1D signal."""
    windows = []
    start = 0
    while start + win_samples <= len(s):
        windows.append(s[start:start + win_samples])
        start += step_samples
    return np.array(windows) if windows else np.array([]).reshape(0, win_samples)


def compute_tau(s, max_lag=None):
    """Compute time delay tau via first zero-crossing of autocorrelation."""
    if max_lag is None:
        max_lag = len(s) // 4
    max_lag = min(max_lag, len(s) - 1)
    sc = s - np.mean(s)
    ac = np.correlate(sc, sc, mode='full')
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-10)
    for i in range(1, min(max_lag, len(ac))):
        if ac[i] <= 0:
            return max(i, 1)
    return max(max_lag // 10, 1)


def takens_embedding(s, dim, tau, subsample=1):
    """Construct a Takens time-delay embedding of signal s."""
    n = len(s) - (dim - 1) * tau
    if n <= 0:
        return np.array([]).reshape(0, dim)
    indices = np.arange(n)[:, None] + np.arange(dim)[None, :] * tau
    pc = s[indices]
    if subsample > 1:
        pc = pc[::subsample]
    return pc


# ============================================================
# TDA FUNCTIONS
# ============================================================

def compute_audio_persistence(point_cloud, max_dim=MAX_DIM, max_edge_length=MAX_EDGE_LENGTH):
    """Compute persistence diagrams from an audio point cloud."""
    if len(point_cloud) < 3:
        return [np.array([[0, 0]]), np.array([[0, 0]])]
    pc_min = point_cloud.min(axis=0)
    pc_range = point_cloud.max(axis=0) - pc_min
    pc_range[pc_range == 0] = 1
    pc_norm = (point_cloud - pc_min) / pc_range
    result = ripser(pc_norm, maxdim=max_dim, thresh=max_edge_length)
    return result["dgms"]


def compute_eeg_persistence(dist_matrix, max_dim=MAX_DIM, max_edge_length=MAX_EDGE_LENGTH):
    """Compute persistence diagrams from an EEG distance matrix."""
    dm = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dm, 0)
    dm = np.maximum(dm, 0)
    result = ripser(dm, maxdim=max_dim, thresh=max_edge_length, distance_matrix=True)
    return result["dgms"]


def extract_features(diagram):
    """Extract scalar features from a persistence diagram."""
    finite_mask = np.isfinite(diagram).all(axis=1)
    fd = diagram[finite_mask]
    n_ess = int(np.sum(~finite_mask))
    if len(fd) == 0:
        return {
            "n_features": 0, "n_essential": n_ess,
            "mean_birth": 0, "std_birth": 0,
            "mean_death": 0, "std_death": 0,
            "mean_persistence": 0, "std_persistence": 0,
            "max_persistence": 0, "total_persistence": 0,
            "persistence_entropy": 0,
        }
    births, deaths = fd[:, 0], fd[:, 1]
    pers = deaths - births
    if len(pers) > 1 and np.sum(pers) > 0:
        pn = pers / np.sum(pers)
        pn = pn[pn > 0]
        ent = -np.sum(pn * np.log(pn + 1e-10)) / np.log(len(pers) + 1e-10)
    else:
        ent = 0
    return {
        "n_features": len(fd), "n_essential": n_ess,
        "mean_birth": float(np.mean(births)),
        "std_birth": float(np.std(births)) if len(births) > 1 else 0,
        "mean_death": float(np.mean(deaths)),
        "std_death": float(np.std(deaths)) if len(deaths) > 1 else 0,
        "mean_persistence": float(np.mean(pers)),
        "std_persistence": float(np.std(pers)) if len(pers) > 1 else 0,
        "max_persistence": float(np.max(pers)),
        "total_persistence": float(np.sum(pers)),
        "persistence_entropy": float(ent),
    }


def safe_wasserstein(dgm1, dgm2):
    """Compute Wasserstein distance between two persistence diagrams, handling edge cases."""
    def clean(d):
        if d.ndim != 2 or d.shape[0] == 0:
            return np.array([[0, 0]])
        m = np.isfinite(d).all(axis=1)
        d = d[m]
        return d if len(d) > 0 else np.array([[0, 0]])
    try:
        return wasserstein_distance(clean(dgm1), clean(dgm2))
    except Exception:
        return np.nan


# ============================================================
# PERMUTATION HELPERS
# ============================================================

def permute_labels_by_subject(y, subjects, rng):
    """
    Permute labels at the subject level (not globally).

    This ensures that all recordings from the same subject receive the
    same permuted label, respecting the group structure used in
    StratifiedGroupKFold cross-validation.
    """
    unique_subjects = np.unique(subjects)
    # Get one label per subject (the true label for that subject)
    subject_labels = np.array([y[subjects == s][0] for s in unique_subjects])
    # Permute at the subject level
    perm_subject_labels = rng.permutation(subject_labels)
    # Map back to all recordings
    y_perm = np.zeros_like(y)
    for s, label in zip(unique_subjects, perm_subject_labels):
        y_perm[subjects == s] = label
    return y_perm
