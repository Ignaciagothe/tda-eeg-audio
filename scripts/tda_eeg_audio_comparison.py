"""
hipotesis 2:
la topologia de eeg es mas similar a la topologia del audio en la condicion lenta que en la condicion rapida.

pipeline:
1.	Audio envolvente → embedding de Takens (por banda) → diagramas (H0, H1)
2.	EEG vs audio: distancia Wasserstein (por ventana y banda)
3.	EEG vs audio: correlación temporal de features TDA
4.	Test dentro de sujeto: Wilcoxon signed-rank

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
import json
import warnings
import sys
import time
from utils import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

BASE_DIR = get_project_root()
DATA_DIR = BASE_DIR / "data"
GRAPHS_DIR = BASE_DIR / "graphs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WINDOW_SEC = 1.0
OVERLAP = 0.75
MAX_WINDOWS = 15  # rendimiento - submuestrear ventanas     
ALPHA = 0.05
N_PERMUTATIONS = 1000

print("comparacion topologia entre señal EEG y señal de audio")

def process_recording(filename, condition):
    mat_path = DATA_DIR / condition / filename
    graph_dir = GRAPHS_DIR / condition / filename.replace('.mat', '')
    if not mat_path.exists() or not graph_dir.exists():
        return None

    subject = filename.split('_')[0]

    audio = load_audio(mat_path)
    audio_rs = resample_audio(audio, FS_AUDIO, FS_EEG)
    envelope = compute_envelope(audio_rs, FS_EEG)

    win_samp = int(WINDOW_SEC * FS_EEG)
    step_samp = int(win_samp * (1 - OVERLAP))

    results = {"filename": filename, "condition": condition,
                "subject": subject, "bands": {}}

    for bname, (lo, hi) in FREQ_BANDS.items():
        audio_band = bandpass_filter(envelope, FS_EEG, lo, hi)
        audio_wins = create_windows(audio_band, win_samp, step_samp)

        dist_file = graph_dir / f"{bname}_distances.npy"
        if not dist_file.exists():
            continue
        eeg_dists = np.load(str(dist_file))

        n_win = min(len(audio_wins), eeg_dists.shape[0])
        if n_win == 0:
            continue

        # Subsample windows evenly
        if n_win > MAX_WINDOWS:
            idx = np.linspace(0, n_win - 1, MAX_WINDOWS, dtype=int)
        else:
            idx = np.arange(n_win)

        # Compute tau from first window
        tau = compute_tau(audio_wins[idx[0]], max_lag=win_samp // 2)

        wass_h0, wass_h1 = [], []
        audio_feat_ts, eeg_feat_ts = [], []

        for w in idx:
            pc = takens_embedding(audio_wins[w], TAKENS_DIM, tau, TAKENS_SUBSAMPLE)
            if len(pc) < 3:
                continue
            a_dgms = compute_audio_persistence(pc)
            e_dgms = compute_eeg_persistence(eeg_dists[w])

            wass_h0.append(safe_wasserstein(e_dgms[0], a_dgms[0]))
            wass_h1.append(safe_wasserstein(e_dgms[1], a_dgms[1]))

            audio_feat_ts.append(extract_features(a_dgms[1]))
            eeg_feat_ts.append(extract_features(e_dgms[1]))

        if not wass_h0:
            continue

        # Temporal correlations
        feat_corrs = {}
        for feat in ["mean_persistence", "total_persistence",
                        "persistence_entropy", "max_persistence", "n_features"]:
            a_ts = [f[feat] for f in audio_feat_ts]
            e_ts = [f[feat] for f in eeg_feat_ts]
            if len(a_ts) >= 5 and np.std(a_ts) > 1e-10 and np.std(e_ts) > 1e-10:
                r, p = spearmanr(a_ts, e_ts)
                feat_corrs[feat] = {"r": float(r), "p": float(p)}
            else:
                feat_corrs[feat] = {"r": 0.0, "p": 1.0}

        results["bands"][bname] = {
            "wasserstein_h0": float(np.nanmean(wass_h0)),
            "wasserstein_h1": float(np.nanmean(wass_h1)),
            "n_windows": len(idx),
            "tau": int(tau),
            "feature_correlations": feat_corrs,
        }

    return results if results["bands"] else None

def run_analysis():
    t0 = time.time()
    print(f" procesar registros (max {MAX_WINDOWS} windows/band)")

    all_results = []
    for condition in ["slow", "fast"]:
        mat_files = sorted([f.name for f in (DATA_DIR / condition).glob("*.mat")])
        print(f"\n{condition.upper()}: {len(mat_files)} recordings")
        for i, fn in enumerate(mat_files):
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                print(f"  [{elapsed:.0f}s] {i+1}/{len(mat_files)}: {fn}")
            r = process_recording(fn, condition)
            if r:
                all_results.append(r)

    elapsed = time.time() - t0
    print(f" se procesaron {len(all_results)} grabaciones válidas ( {elapsed:.0f}s)")

    rows = []
    for r in all_results:
        for bname, bd in r["bands"].items():
            row = {"filename": r["filename"], "condition": r["condition"],
                   "subject": r["subject"], "band": bname,
                   "wasserstein_h0": bd["wasserstein_h0"],
                   "wasserstein_h1": bd["wasserstein_h1"],
                   "n_windows": bd["n_windows"], "tau": bd["tau"]}
            for feat, vals in bd["feature_correlations"].items():
                row[f"corr_{feat}_r"] = vals["r"]
                row[f"corr_{feat}_p"] = vals["p"]
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"DataFrame: {df.shape[0]} rows, {df['subject'].nunique()} subjects")


    print("pruebas estadísticas")
    stats = {}
    for band in FREQ_BANDS:
        bdf = df[df["band"] == band]
        sm = bdf.groupby(["subject", "condition"]).agg({
            "wasserstein_h0": "mean", "wasserstein_h1": "mean",
            "corr_mean_persistence_r": "mean",
            "corr_persistence_entropy_r": "mean",
        }).reset_index()

        slow = sm[sm["condition"] == "slow"]
        fast = sm[sm["condition"] == "fast"]
        common = set(slow["subject"]) & set(fast["subject"])
        slow = slow[slow["subject"].isin(common)].sort_values("subject")
        fast = fast[fast["subject"].isin(common)].sort_values("subject")
        n = len(common)
        bs = {"n_subjects": n, "band": band}

        if n >= 5:
            d0 = slow["wasserstein_h0"].values - fast["wasserstein_h0"].values
            d1 = slow["wasserstein_h1"].values - fast["wasserstein_h1"].values
            dc = slow["corr_mean_persistence_r"].values - fast["corr_mean_persistence_r"].values

            _, p0 = wilcoxon(d0) if np.any(d0 != 0) else (0, 1.0)
            _, p1 = wilcoxon(d1) if np.any(d1 != 0) else (0, 1.0)
            _, pc = wilcoxon(dc) if np.any(dc != 0) else (0, 1.0)

            # Permutation test H1 (sign-flip)
            perm_rng = np.random.default_rng(42)
            obs = np.mean(d1)
            exceed = sum(1 for _ in range(N_PERMUTATIONS)
                         if abs(np.mean(d1 * perm_rng.choice([-1, 1], n))) >= abs(obs))
            perm_p = (exceed + 1) / (N_PERMUTATIONS + 1)

            # Effect size (Cohen's d with sample std)
            cohens_d = np.mean(d1) / (np.std(d1, ddof=1) + 1e-10)

            bs.update({
                "wass_h0_slow": float(slow["wasserstein_h0"].mean()),
                "wass_h0_fast": float(fast["wasserstein_h0"].mean()),
                "wass_h0_p": float(p0),
                "wass_h1_slow": float(slow["wasserstein_h1"].mean()),
                "wass_h1_fast": float(fast["wasserstein_h1"].mean()),
                "wass_h1_p": float(p1),
                "wass_h1_perm_p": float(perm_p),
                "wass_h1_cohens_d": float(cohens_d),
                "wass_h1_direction": "slow < fast" if np.mean(d1) < 0 else "slow > fast",
                "corr_slow": float(slow["corr_mean_persistence_r"].mean()),
                "corr_fast": float(fast["corr_mean_persistence_r"].mean()),
                "corr_p": float(pc),
                "n_slow_lower": int(np.sum(d1 < 0)),
            })
        stats[band] = bs

    # FDR correction
    pvals = [stats[b].get("wass_h1_p", 1.0) for b in FREQ_BANDS]
    reject, pfdr, _, _ = multipletests(pvals, alpha=ALPHA, method='fdr_bh')
    for i, band in enumerate(FREQ_BANDS):
        stats[band]["wass_h1_p_fdr"] = float(pfdr[i])
        stats[band]["wass_h1_sig_fdr"] = bool(reject[i])

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Wasserstein Distance (EEG-Audio)")
    print("=" * 60)
    for band, s in stats.items():
        if "wass_h1_slow" in s:
            sig = "***" if s.get("wass_h1_sig_fdr") else ("*" if s["wass_h1_p"] < 0.05 else "ns")
            print(f"\n{band.upper()} (n={s['n_subjects']}):")
            print(f"  W_H1: slow={s['wass_h1_slow']:.4f} fast={s['wass_h1_fast']:.4f} "
                  f"p={s['wass_h1_p']:.4f} p_fdr={s['wass_h1_p_fdr']:.4f} {sig}")
            print(f"  W_H0: slow={s['wass_h0_slow']:.4f} fast={s['wass_h0_fast']:.4f} p={s['wass_h0_p']:.4f}")
            print(f"  Direction: {s['wass_h1_direction']} | Cohen's d={s['wass_h1_cohens_d']:.3f}")
            print(f"  Permutation p={s['wass_h1_perm_p']:.4f}")
            print(f"  Feature corr (mean_pers): slow={s['corr_slow']:.4f} fast={s['corr_fast']:.4f} p={s['corr_p']:.4f}")


    print("plots")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, band in enumerate(FREQ_BANDS):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]
        bdf = df[df["band"] == band]
        sm = bdf.groupby(["subject", "condition"])["wasserstein_h1"].mean().reset_index()
        sv = sm[sm["condition"] == "slow"]["wasserstein_h1"].values
        fv = sm[sm["condition"] == "fast"]["wasserstein_h1"].values

        bp = ax.boxplot([sv, fv], positions=[0, 1], widths=0.6, patch_artist=True,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        bp['boxes'][0].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_facecolor('#FF6B6B')
        pf = stats[band].get("wass_h1_p_fdr", 1.0)
        sig = "***" if pf < 0.001 else ("**" if pf < 0.01 else ("*" if pf < 0.05 else "ns"))
        ax.set_title(f"{band.upper()} (p_fdr={pf:.3f}) {sig}", fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Slow", "Fast"])
        ax.set_ylabel("Wasserstein H1")
        ax.grid(True, alpha=0.3)


    ax = axes[1, 2]
    bands_list = list(FREQ_BANDS.keys())
    sl = [stats[b].get("wass_h1_slow", 0) for b in bands_list]
    ft = [stats[b].get("wass_h1_fast", 0) for b in bands_list]
    x = np.arange(len(bands_list))
    ax.bar(x - 0.175, sl, 0.35, label='Slow', color='#4ECDC4', alpha=0.8)
    ax.bar(x + 0.175, ft, 0.35, label='Fast', color='#FF6B6B', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands_list])
    ax.set_ylabel("Mean Wasserstein H1")
    ax.set_title("Summary by Band", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("EEG-Audio Topological Comparison (Wasserstein H1)\n"
                 "Lower = brain topology more similar to audio topology",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eeg_audio_tda_comparison.png", dpi=200, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / "eeg_audio_tda_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved: eeg_audio_tda_comparison.png/pdf")


    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    for idx, feat in enumerate(["corr_mean_persistence_r", "corr_persistence_entropy_r"]):
        ax = axes2[idx]
        fl = feat.replace("corr_", "").replace("_r", "").replace("_", " ").title()
        for band in FREQ_BANDS:
            bd_s = df[(df["band"] == band) & (df["condition"] == "slow")][feat].mean()
            bd_f = df[(df["band"] == band) & (df["condition"] == "fast")][feat].mean()
            ax.scatter([band], [bd_s], color='#4ECDC4', s=100, zorder=5)
            ax.scatter([band], [bd_f], color='#FF6B6B', s=100, zorder=5)
            ax.plot([band, band], [bd_s, bd_f], 'k-', alpha=0.3)
        ax.axhline(0, color='grey', ls='--', alpha=0.5)
        ax.set_ylabel("Spearman r (EEG-Audio)")
        ax.set_title(f"Temporal Correlation: {fl}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.suptitle("EEG-Audio TDA Feature Temporal Correlation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eeg_audio_tda_temporal_correlation.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: eeg_audio_tda_temporal_correlation.png")

    # ---- Save results ----
    print("\nPHASE 4: Saving")
    output = {
        "analysis": "EEG-Audio Topological Comparison",
        "method": "Wasserstein distance on persistence diagrams + temporal feature correlation",
        "audio_construction": f"Takens embedding (dim={TAKENS_DIM}, tau=auto, subsample={TAKENS_SUBSAMPLE})",
        "eeg_construction": "Connectivity graph distance matrix (existing pipeline)",
        "n_recordings": len(all_results),
        "n_subjects": df["subject"].nunique(),
        "n_slow": len([r for r in all_results if r["condition"] == "slow"]),
        "n_fast": len([r for r in all_results if r["condition"] == "fast"]),
        "max_windows_per_recording": MAX_WINDOWS,
        "statistical_test": "Wilcoxon signed-rank (within-subject, paired)",
        "multiple_comparison": "Benjamini-Hochberg FDR",
        "band_results": stats,
    }
    with open(RESULTS_DIR / "eeg_audio_tda_comparison.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    df.to_csv(RESULTS_DIR / "eeg_audio_tda_detailed.csv", index=False)
    print("  Saved: eeg_audio_tda_comparison.json + eeg_audio_tda_detailed.csv")

    print('resumen')
    any_sig = any(stats[b].get("wass_h1_sig_fdr", False) for b in FREQ_BANDS)
    if any_sig:
        sb = [b for b in FREQ_BANDS if stats[b].get("wass_h1_sig_fdr")]
        print(f"diferencias significativas en: {sb}")
    else:
        any_unc = any(stats[b].get("wass_h1_p", 1) < 0.05 for b in FREQ_BANDS)
        if any_unc:
            ub = [b for b in FREQ_BANDS if stats[b].get("wass_h1_p", 1) < 0.05]
            print(f"Tendencia (no corregida) en: {ub}")
        else:
            print("no hay diferencias significativas encontradas entre condiciones")


    for band in FREQ_BANDS:
        s = stats[band]
        if "wass_h1_slow" in s:
            d = "closer" if s["wass_h1_slow"] < s["wass_h1_fast"] else "farther"
            print(f"  {band}: slow EEG topologically {d} to audio "
                  f"(W_slow={s['wass_h1_slow']:.4f} W_fast={s['wass_h1_fast']:.4f})")

    print(f"\nTiempo total: {time.time() - t0:.0f}s")
    return output


if __name__ == "__main__":
    run_analysis()
