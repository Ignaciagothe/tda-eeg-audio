"""
Matched vs Mismatched Wasserstein Control
==========================================
Tests whether EEG-audio topological coupling is genuine by comparing:
  - MATCHED:    W(EEG_slow, Audio_slow)  and  W(EEG_fast, Audio_fast)
  - MISMATCHED: W(EEG_slow, Audio_fast)  and  W(EEG_fast, Audio_slow)

If coupling is genuine: matched < mismatched in both conditions.
If the effect is just "audio slow ≠ audio fast": no difference.

Strategy: for each subject, for each band, compute:
  1. matched_slow   = mean W(EEG_slow_windows, Audio_slow_windows) [already computed]
  2. matched_fast   = mean W(EEG_fast_windows, Audio_fast_windows) [already computed]
  3. mismatched_slow = mean W(EEG_slow_windows, Audio_fast_windows) [NEW]
  4. mismatched_fast = mean W(EEG_fast_windows, Audio_slow_windows) [NEW]

For mismatched, we pair each EEG recording with a random audio recording
from the OPPOSITE condition (same subject if available, otherwise random).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sig_proc
from scipy import io as sio
from scipy.stats import wilcoxon
from ripser import ripser
from persim import wasserstein as wasserstein_distance
from statsmodels.stats.multitest import multipletests
import json, warnings, time, sys

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = Path("/sessions/modest-adoring-brown/mnt/tda-eeg-audio")
DATA_DIR = BASE / "data"
GRAPHS_DIR = BASE / "graphs"
RESULTS_DIR = BASE / "results"

FREQ_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
              "beta": (13, 30), "gamma": (30, 50)}
FS_AUDIO = 44100; FS_EEG = 250
WINDOW_SEC = 1.0; OVERLAP = 0.75
MAX_DIM = 1; MAX_EDGE_LENGTH = 2.0
TAKENS_DIM = 3; TAKENS_SUBSAMPLE = 2
MAX_WINDOWS = 15

# ── Signal processing (copied from original pipeline) ──
def load_audio(mat_path):
    mat = sio.loadmat(str(mat_path))
    y = mat['y']
    if y.ndim == 2: y = y.mean(axis=1)
    return y.astype(np.float64)

def compute_envelope(s, fs):
    analytic = sig_proc.hilbert(s)
    env = np.abs(analytic)
    nyq = fs / 2
    cutoff = min(50, nyq * 0.9)
    b, a = sig_proc.butter(4, cutoff / nyq, btype='low')
    return sig_proc.filtfilt(b, a, env)

def bandpass_filter(s, fs, low, high):
    nyq = fs / 2
    lo = max(low / nyq, 0.001); hi = min(high / nyq, 0.999)
    if lo >= hi: return s
    b, a = sig_proc.butter(4, [lo, hi], btype='band')
    return sig_proc.filtfilt(b, a, s)

def create_windows(s, win_samples, step_samples):
    windows = []
    start = 0
    while start + win_samples <= len(s):
        windows.append(s[start:start + win_samples])
        start += step_samples
    return np.array(windows) if windows else np.array([]).reshape(0, win_samples)

def compute_tau(s, max_lag=None):
    if max_lag is None: max_lag = len(s) // 4
    max_lag = min(max_lag, len(s) - 1)
    sc = s - np.mean(s)
    ac = np.correlate(sc, sc, mode='full')
    ac = ac[len(ac) // 2:]
    ac = ac / (ac[0] + 1e-10)
    for i in range(1, min(max_lag, len(ac))):
        if ac[i] <= 0: return max(i, 1)
    return max(max_lag // 10, 1)

def takens_embedding(s, dim, tau, subsample=1):
    n = len(s) - (dim - 1) * tau
    if n <= 0: return np.array([]).reshape(0, dim)
    indices = np.arange(n)[:, None] + np.arange(dim)[None, :] * tau
    pc = s[indices]
    if subsample > 1: pc = pc[::subsample]
    return pc

def compute_audio_persistence(pc):
    if len(pc) < 3: return [np.array([[0, 0]]), np.array([[0, 0]])]
    pc_min = pc.min(axis=0); pc_range = pc.max(axis=0) - pc_min
    pc_range[pc_range == 0] = 1
    pc_norm = (pc - pc_min) / pc_range
    return ripser(pc_norm, maxdim=MAX_DIM, thresh=MAX_EDGE_LENGTH)["dgms"]

def compute_eeg_persistence(dm):
    dm = (dm + dm.T) / 2; np.fill_diagonal(dm, 0); dm = np.maximum(dm, 0)
    return ripser(dm, maxdim=MAX_DIM, thresh=MAX_EDGE_LENGTH, distance_matrix=True)["dgms"]

def safe_wasserstein(dgm1, dgm2):
    def clean(d):
        if d.ndim != 2 or d.shape[0] == 0: return np.array([[0, 0]])
        m = np.isfinite(d).all(axis=1); d = d[m]
        return d if len(d) > 0 else np.array([[0, 0]])
    try: return wasserstein_distance(clean(dgm1), clean(dgm2))
    except: return np.nan

# ── Precompute audio persistence diagrams for all recordings ──
def get_audio_diagrams(filename, condition):
    """Get per-band, per-window audio persistence diagrams for one recording."""
    mat_path = DATA_DIR / condition / filename
    if not mat_path.exists(): return None
    try:
        audio = load_audio(mat_path)
        audio_dec = sig_proc.decimate(audio, 10, zero_phase=True)
        audio_dec = sig_proc.decimate(audio_dec, 10, zero_phase=True)
        n_target = int(len(audio) * FS_EEG / FS_AUDIO)
        audio_rs = sig_proc.resample(audio_dec, n_target)
        envelope = compute_envelope(audio_rs, FS_EEG)
        win_samp = int(WINDOW_SEC * FS_EEG)
        step_samp = int(win_samp * (1 - OVERLAP))

        result = {}
        for bname, (lo, hi) in FREQ_BANDS.items():
            audio_band = bandpass_filter(envelope, FS_EEG, lo, hi)
            audio_wins = create_windows(audio_band, win_samp, step_samp)
            n_win = len(audio_wins)
            if n_win == 0: continue
            if n_win > MAX_WINDOWS:
                idx = np.linspace(0, n_win - 1, MAX_WINDOWS, dtype=int)
            else:
                idx = np.arange(n_win)
            tau = compute_tau(audio_wins[idx[0]], max_lag=win_samp // 2)
            dgms_list = []
            for w in idx:
                pc = takens_embedding(audio_wins[w], TAKENS_DIM, tau, TAKENS_SUBSAMPLE)
                if len(pc) < 3: continue
                dgms_list.append(compute_audio_persistence(pc))
            result[bname] = dgms_list
        return result
    except Exception as e:
        print(f"  Audio error {filename}: {e}", file=sys.stderr)
        return None

def get_eeg_diagrams(filename, condition):
    """Get per-band, per-window EEG persistence diagrams from cached distance matrices."""
    graph_dir = GRAPHS_DIR / condition / filename.replace('.mat', '')
    if not graph_dir.exists(): return None
    result = {}
    for bname in FREQ_BANDS:
        dist_file = graph_dir / f"{bname}_distances.npy"
        if not dist_file.exists(): continue
        eeg_dists = np.load(str(dist_file))
        n_win = eeg_dists.shape[0]
        if n_win == 0: continue
        if n_win > MAX_WINDOWS:
            idx = np.linspace(0, n_win - 1, MAX_WINDOWS, dtype=int)
        else:
            idx = np.arange(n_win)
        dgms_list = []
        for w in idx:
            dgms_list.append(compute_eeg_persistence(eeg_dists[w]))
        result[bname] = dgms_list
    return result

def compute_cross_wasserstein(eeg_dgms_band, audio_dgms_band):
    """Compute mean W_H1 between two lists of persistence diagrams (matched by window index)."""
    n = min(len(eeg_dgms_band), len(audio_dgms_band))
    if n == 0: return np.nan
    vals = []
    for i in range(n):
        w = safe_wasserstein(eeg_dgms_band[i][1], audio_dgms_band[i][1])
        vals.append(w)
    return np.nanmean(vals)

# ── Main ──
def main():
    t0 = time.time()
    print("=" * 60)
    print("MATCHED vs MISMATCHED WASSERSTEIN CONTROL")
    print("=" * 60)

    # Build subject -> files mapping
    subj_files = {"slow": {}, "fast": {}}
    for cond in ["slow", "fast"]:
        for f in sorted((DATA_DIR / cond).glob("*.mat")):
            subj = f.stem.split('_')[0]
            subj_files[cond].setdefault(subj, []).append(f.name)

    # Get subjects present in both conditions
    common_subjects = sorted(set(subj_files["slow"].keys()) & set(subj_files["fast"].keys()))
    print(f"Subjects in both conditions: {len(common_subjects)}")

    # For mismatched: pair each recording with a random recording from opposite condition (same subject)
    rows = []
    n_done = 0
    n_total = sum(len(subj_files["slow"].get(s, [])) + len(subj_files["fast"].get(s, []))
                  for s in common_subjects)

    for si, subj in enumerate(common_subjects):
        slow_files = subj_files["slow"].get(subj, [])
        fast_files = subj_files["fast"].get(subj, [])
        if not slow_files or not fast_files:
            continue

        # Pick one representative file per condition for mismatched audio
        mismatch_fast_audio_file = fast_files[0]  # use first fast file as mismatch audio for slow EEG
        mismatch_slow_audio_file = slow_files[0]  # use first slow file as mismatch audio for fast EEG

        # Precompute mismatched audio diagrams once per subject
        mismatch_audio_for_slow_eeg = get_audio_diagrams(mismatch_fast_audio_file, "fast")
        mismatch_audio_for_fast_eeg = get_audio_diagrams(mismatch_slow_audio_file, "slow")

        # Process slow recordings
        for fn in slow_files:
            matched_audio = get_audio_diagrams(fn, "slow")
            eeg = get_eeg_diagrams(fn, "slow")
            if eeg is None: continue

            for bname in FREQ_BANDS:
                if bname not in eeg: continue
                eeg_b = eeg[bname]

                # Matched: W(EEG_slow, Audio_slow_same_recording)
                w_matched = np.nan
                if matched_audio and bname in matched_audio:
                    w_matched = compute_cross_wasserstein(eeg_b, matched_audio[bname])

                # Mismatched: W(EEG_slow, Audio_fast_same_subject)
                w_mismatched = np.nan
                if mismatch_audio_for_slow_eeg and bname in mismatch_audio_for_slow_eeg:
                    w_mismatched = compute_cross_wasserstein(eeg_b, mismatch_audio_for_slow_eeg[bname])

                rows.append({
                    "subject": subj, "condition": "slow", "band": bname,
                    "w_matched": w_matched, "w_mismatched": w_mismatched
                })
            n_done += 1

        # Process fast recordings
        for fn in fast_files:
            matched_audio = get_audio_diagrams(fn, "fast")
            eeg = get_eeg_diagrams(fn, "fast")
            if eeg is None: continue

            for bname in FREQ_BANDS:
                if bname not in eeg: continue
                eeg_b = eeg[bname]

                w_matched = np.nan
                if matched_audio and bname in matched_audio:
                    w_matched = compute_cross_wasserstein(eeg_b, matched_audio[bname])

                w_mismatched = np.nan
                if mismatch_audio_for_fast_eeg and bname in mismatch_audio_for_fast_eeg:
                    w_mismatched = compute_cross_wasserstein(eeg_b, mismatch_audio_for_fast_eeg[bname])

                rows.append({
                    "subject": subj, "condition": "fast", "band": bname,
                    "w_matched": w_matched, "w_mismatched": w_mismatched
                })
            n_done += 1

        if (si + 1) % 5 == 0 or si == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] Subject {si+1}/{len(common_subjects)} ({subj})")

    df = pd.DataFrame(rows)
    print(f"\nDataFrame: {len(df)} rows, {df['subject'].nunique()} subjects")
    print(f"Total time: {time.time() - t0:.0f}s")

    # ── Statistical tests ──
    print("\n" + "=" * 60)
    print("RESULTS: Matched vs Mismatched")
    print("=" * 60)

    results = {}
    for band in FREQ_BANDS:
        bdf = df[df["band"] == band].dropna(subset=["w_matched", "w_mismatched"])
        # Aggregate to subject level
        sm = bdf.groupby(["subject"]).agg(
            w_matched=("w_matched", "mean"),
            w_mismatched=("w_mismatched", "mean")
        ).dropna()
        n = len(sm)
        if n < 5:
            results[band] = {"n": n, "status": "insufficient"}
            continue

        diff = sm["w_matched"].values - sm["w_mismatched"].values
        mean_matched = sm["w_matched"].mean()
        mean_mismatched = sm["w_mismatched"].mean()

        if np.any(diff != 0):
            stat, p = wilcoxon(diff)
        else:
            p = 1.0

        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
        n_matched_lower = int(np.sum(diff < 0))

        results[band] = {
            "n": n,
            "w_matched": float(mean_matched),
            "w_mismatched": float(mean_mismatched),
            "direction": "matched < mismatched" if mean_matched < mean_mismatched else "matched > mismatched",
            "p": float(p),
            "cohens_d": float(cohens_d),
            "n_matched_lower": n_matched_lower,
            "pct_matched_lower": float(n_matched_lower / n * 100)
        }

    # FDR correction
    pvals = [results[b].get("p", 1.0) for b in FREQ_BANDS]
    reject, pfdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for i, band in enumerate(FREQ_BANDS):
        if "p" in results[band]:
            results[band]["p_fdr"] = float(pfdr[i])
            results[band]["sig_fdr"] = bool(reject[i])

    # Print
    for band, r in results.items():
        if "w_matched" not in r: continue
        sig = "***" if r.get("p_fdr", 1) < 0.001 else ("**" if r.get("p_fdr", 1) < 0.01 else ("*" if r.get("p_fdr", 1) < 0.05 else "ns"))
        print(f"\n{band.upper()} (n={r['n']}):")
        print(f"  Matched:    {r['w_matched']:.4f}")
        print(f"  Mismatched: {r['w_mismatched']:.4f}")
        print(f"  Direction:  {r['direction']}")
        print(f"  p={r['p']:.4f}  p_fdr={r.get('p_fdr', 'N/A')}  d={r['cohens_d']:.3f}  {sig}")
        print(f"  Subjects matched<mismatched: {r['n_matched_lower']}/{r['n']} ({r['pct_matched_lower']:.0f}%)")

    # Also compute per-condition breakdown
    print("\n" + "=" * 60)
    print("PER-CONDITION BREAKDOWN")
    print("=" * 60)
    for band in FREQ_BANDS:
        bdf = df[df["band"] == band].dropna(subset=["w_matched", "w_mismatched"])
        for cond in ["slow", "fast"]:
            cdf = bdf[bdf["condition"] == cond]
            sm = cdf.groupby("subject").agg(
                w_matched=("w_matched", "mean"),
                w_mismatched=("w_mismatched", "mean")
            ).dropna()
            if len(sm) < 5: continue
            diff = sm["w_matched"].values - sm["w_mismatched"].values
            if np.any(diff != 0):
                _, p = wilcoxon(diff)
            else:
                p = 1.0
            d = np.mean(diff) / (np.std(diff) + 1e-10)
            n_lower = int(np.sum(diff < 0))
            print(f"  {band.upper()} {cond}: matched={sm['w_matched'].mean():.4f} "
                  f"mismatch={sm['w_mismatched'].mean():.4f} p={p:.4f} d={d:.3f} "
                  f"({n_lower}/{len(sm)} matched<mismatch)")

    # Save
    output = {
        "analysis": "Matched vs Mismatched Wasserstein Control",
        "description": "Tests whether W(EEG, Audio_same_condition) < W(EEG, Audio_opposite_condition)",
        "results_by_band": results
    }
    with open(RESULTS_DIR / "matched_vs_mismatched.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_DIR / 'matched_vs_mismatched.json'}")

if __name__ == "__main__":
    main()
