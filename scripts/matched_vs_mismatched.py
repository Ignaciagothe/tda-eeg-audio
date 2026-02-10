"""
to test the validity of EEG-audio topological coupling results, we compare :
  - matched:   wasserstein distance between EEG_slow and Audio_slow and EEG_fast with Audio_fast
  - mismatched: wasserstein distance between EEG_slow and Audio_fast and EEG_fast with Audio_slow

we should see matched < mismatched in both conditions. (for mismatched, use audio from the opposite condition from the same subject.)
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import json, warnings, time, sys

from utils import (
    get_project_root, load_audio, compute_envelope, bandpass_filter,
    resample_audio, create_windows, compute_tau, takens_embedding,
    compute_audio_persistence, compute_eeg_persistence, safe_wasserstein,
    FREQ_BANDS, FS_AUDIO, FS_EEG, TAKENS_DIM, TAKENS_SUBSAMPLE
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

BASE = get_project_root()
DATA_DIR = BASE / "data"
GRAPHS_DIR = BASE / "graphs"
RESULTS_DIR = BASE / "results"

WINDOW_SEC = 1.0
OVERLAP = 0.75
MAX_WINDOWS = 15


def get_audio_diagrams(filename, condition):
    """get diagramas persistencia para cada banda freq y cada window de un audio."""
    mat_path = DATA_DIR / condition / filename
    if not mat_path.exists(): return None

    audio = load_audio(mat_path)
    audio_rs = resample_audio(audio, FS_AUDIO, FS_EEG)
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


def get_eeg_diagrams(filename, condition):
    """diagrama de presistencia para eeg para cada banda y ventana de matriz distancia """
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
    """Calcular el promedio de W_H1 entre dos listas de diagramas de persistencia emparejados por idx ventana"""
    n = min(len(eeg_dgms_band), len(audio_dgms_band))
    if n == 0: return np.nan
    vals = []
    for i in range(n):
        w = safe_wasserstein(eeg_dgms_band[i][1], audio_dgms_band[i][1])
        vals.append(w)
    return np.nanmean(vals)

def main():
    t0 = time.time()
    subj_files = {"slow": {}, "fast": {}}
    for cond in ["slow", "fast"]:
        for f in sorted((DATA_DIR / cond).glob("*.mat")):
            subj = f.stem.split('_')[0]
            subj_files[cond].setdefault(subj, []).append(f.name)

    common_subjects = sorted(set(subj_files["slow"].keys()) & set(subj_files["fast"].keys()))
    print(f"Subjects in both conditions: {len(common_subjects)}")

    rows = []
    n_done = 0

    for si, subj in enumerate(common_subjects):
        slow_files = subj_files["slow"].get(subj, [])
        fast_files = subj_files["fast"].get(subj, [])
        if not slow_files or not fast_files:
            continue

        mismatch_fast_audio_file = fast_files[0]
        mismatch_slow_audio_file = slow_files[0]

        mismatch_audio_for_slow_eeg = get_audio_diagrams(mismatch_fast_audio_file, "fast")
        mismatch_audio_for_fast_eeg = get_audio_diagrams(mismatch_slow_audio_file, "slow")

        for fn in slow_files:
            matched_audio = get_audio_diagrams(fn, "slow")
            eeg = get_eeg_diagrams(fn, "slow")
            if eeg is None:
                continue

            for bname in FREQ_BANDS:
                if bname not in eeg:
                    continue
                eeg_b = eeg[bname]

                w_matched = np.nan
                if matched_audio and bname in matched_audio:
                    w_matched = compute_cross_wasserstein(eeg_b, matched_audio[bname])

                w_mismatched = np.nan
                if mismatch_audio_for_slow_eeg and bname in mismatch_audio_for_slow_eeg:
                    w_mismatched = compute_cross_wasserstein(eeg_b, mismatch_audio_for_slow_eeg[bname])

                rows.append({
                    "subject": subj, "condition": "slow", "band": bname,
                    "w_matched": w_matched, "w_mismatched": w_mismatched
                })
            n_done += 1

        for fn in fast_files:
            matched_audio = get_audio_diagrams(fn, "fast")
            eeg = get_eeg_diagrams(fn, "fast")
            if eeg is None:
                continue

            for bname in FREQ_BANDS:
                if bname not in eeg:
                    continue
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
            print(f"[{elapsed:.0f}s] {si+1}/{len(common_subjects)} subjects")

    df = pd.DataFrame(rows)
    print(f"Rows: {len(df)} | Subjects: {df['subject'].nunique()} | Time: {time.time() - t0:.0f}s")

    results = {}
    for band in FREQ_BANDS:
        bdf = df[df["band"] == band].dropna(subset=["w_matched", "w_mismatched"])
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
            _, p = wilcoxon(diff)
        else:
            p = 1.0

        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
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

    pvals = [results[b].get("p", 1.0) for b in FREQ_BANDS]
    reject, pfdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for i, band in enumerate(FREQ_BANDS):
        if "p" in results[band]:
            results[band]["p_fdr"] = float(pfdr[i])
            results[band]["sig_fdr"] = bool(reject[i])

    for band, r in results.items():
        if "w_matched" not in r:
            continue
        sig = "***" if r.get("p_fdr", 1) < 0.001 else ("**" if r.get("p_fdr", 1) < 0.01 else ("*" if r.get("p_fdr", 1) < 0.05 else "ns"))
        print(f"{band.upper()} n={r['n']} | matched={r['w_matched']:.4f} | mismatch={r['w_mismatched']:.4f} | "
              f"p={r['p']:.4f} | p_fdr={r.get('p_fdr', 'N/A')} | d={r['cohens_d']:.3f} | {sig} | "
              f"{r['n_matched_lower']}/{r['n']} matched<mismatch")


    for band in FREQ_BANDS:
        bdf = df[df["band"] == band].dropna(subset=["w_matched", "w_mismatched"])
        for cond in ["slow", "fast"]:
            cdf = bdf[bdf["condition"] == cond]
            sm = cdf.groupby("subject").agg(
                w_matched=("w_matched", "mean"),
                w_mismatched=("w_mismatched", "mean")
            ).dropna()
            if len(sm) < 5:
                continue

            diff = sm["w_matched"].values - sm["w_mismatched"].values
            if np.any(diff != 0):
                _, p = wilcoxon(diff)
            else:
                p = 1.0

            d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
            n_lower = int(np.sum(diff < 0))

            print(f"{band.upper()} {cond} | matched={sm['w_matched'].mean():.4f} | mismatch={sm['w_mismatched'].mean():.4f} | "
                  f"p={p:.4f} | d={d:.3f} | {n_lower}/{len(sm)} matched<mismatch")

    out_path = RESULTS_DIR / "matched_vs_mismatched.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
   