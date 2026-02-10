"""
version 2
Clasificacion de eeg lento vs rapido usando tda features guardadas  
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, StratifiedGroupKFold, GroupKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.patches import Patch
from utils import permute_labels_by_subject

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE / "features"
RESULTS_DIR = BASE / "results"
FIGURES_DIR = BASE / "paper" / "figures"
OLD_RESULTS = RESULTS_DIR / "old_classification"
OLD_RESULTS.mkdir(exist_ok=True)

# ── Load cached features ──
print("Loading cached features...")
X = np.load(FEATURES_DIR / "X.npy")
y = np.load(FEATURES_DIR / "y.npy")
subjects = np.load(FEATURES_DIR / "subjects.npy", allow_pickle=True)

with open(FEATURES_DIR / "feature_names.txt") as f:
    feature_names = [line.strip() for line in f if line.strip()]

print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Subjects: {len(np.unique(subjects))}")
print(f"  Slow: {np.sum(y == 0)}, Fast: {np.sum(y == 1)}")


nan_mask = np.isnan(X).any(axis=1)
inf_mask = np.isinf(X).any(axis=1)
valid_mask = ~(nan_mask | inf_mask)
n_removed = (~valid_mask).sum()
if n_removed > 0:
    print(f"  Removing {n_removed} samples with NaN/Inf")
    X = X[valid_mask]
    y = y[valid_mask]
    subjects = subjects[valid_mask]

N_SPLITS = 5
N_PERMUTATIONS = 1000
RANDOM_STATE = 42

try:
    gkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_name = "StratifiedGroupKFold"
except Exception:
    gkf = GroupKFold(n_splits=N_SPLITS)
    cv_name = "GroupKFold"
print(f"CV method: {cv_name}")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

cv_scores = cross_val_score(pipeline, X, y, groups=subjects, cv=gkf, scoring="accuracy")
print(f"  Per-fold accuracy: {cv_scores}")
print(f"  Mean accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_pred_cv = cross_val_predict(pipeline, X, y, groups=subjects, cv=gkf)
cv_f1 = f1_score(y, y_pred_cv, average="weighted")
print(f"  F1 score (weighted): {cv_f1:.4f}")

y_proba_cv = cross_val_predict(pipeline, X, y, groups=subjects, cv=gkf, method="predict_proba")
cv_auc = roc_auc_score(y, y_proba_cv[:, 1])
print(f"  ROC-AUC: {cv_auc:.4f}")

cm = confusion_matrix(y, y_pred_cv)
print(f"  Confusion matrix:\n{cm}")


slow_correct = cm[0, 0] / cm[0].sum() * 100
fast_correct = cm[1, 1] / cm[1].sum() * 100
print(f"  Slow correctly classified: {slow_correct:.1f}%")
print(f"  Fast correctly classified: {fast_correct:.1f}%")



pipeline.fit(X, y)
importances = pipeline.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

importance_df["band"] = importance_df["feature"].apply(
    lambda x: x.split("_")[0] if "_" in x else "unknown"
)
importance_df["dimension"] = importance_df["feature"].apply(
    lambda x: "H0" if "_h0_" in x else "H1" if "_h1_" in x else "unknown"
)

print("\nTop 15 features:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

band_importance = importance_df.groupby("band")["importance"].sum().sort_values(ascending=False)
total_imp = band_importance.sum()
print("\nImportance by band:")
for band, imp in band_importance.items():
    print(f"  {band}: {imp:.4f} ({imp/total_imp*100:.1f}%)")

dim_importance = importance_df.groupby("dimension")["importance"].sum()
print("\nImportance by dimension:")
for dim, imp in dim_importance.items():
    print(f"  {dim}: {imp:.4f} ({imp/dim_importance.sum()*100:.1f}%)")


observed_acc = cv_scores.mean()
rng = np.random.RandomState(RANDOM_STATE)
null_dist = []

for i in tqdm(range(N_PERMUTATIONS), desc="Permutation test"):
    y_perm = permute_labels_by_subject(y, subjects, rng)
    perm_acc = cross_val_score(
        pipeline, X, y_perm, groups=subjects, cv=gkf, scoring="accuracy"
    ).mean()
    null_dist.append(perm_acc)

null_dist = np.array(null_dist)
p_value = (np.sum(null_dist >= observed_acc) + 1) / (N_PERMUTATIONS + 1)
effect_size = (observed_acc - null_dist.mean()) / null_dist.std()

print(f"  Observed accuracy: {observed_acc:.4f}")
print(f"  Null distribution mean: {null_dist.mean():.4f} ± {null_dist.std():.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Cohen's d: {effect_size:.2f}")


unique_subjects = np.unique(subjects)
subject_acc = {}
for subj in unique_subjects:
    mask = subjects == subj
    subj_correct = (y_pred_cv[mask] == y[mask]).mean()
    subject_acc[subj] = subj_correct

subject_acc_arr = np.array([subject_acc[s] for s in unique_subjects])
n_subj = len(unique_subjects)


boot_rng = np.random.default_rng(RANDOM_STATE)
boot_accs = []
for _ in range(2000):
    boot_idx = boot_rng.choice(n_subj, size=n_subj, replace=True)
    boot_accs.append(subject_acc_arr[boot_idx].mean())

boot_accs = np.array(boot_accs)
ci_lower = np.percentile(boot_accs, 2.5)
ci_upper = np.percentile(boot_accs, 97.5)
print(f"  Bootstrap mean: {boot_accs.mean():.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")

fold_ci_lower = observed_acc - 1.96 * cv_scores.std()
fold_ci_upper = observed_acc + 1.96 * cv_scores.std()
print(f"  Fold-based CI (normal approx): [{fold_ci_lower:.4f}, {fold_ci_upper:.4f}]")

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Slow", "Fast"],
    yticklabels=["Slow", "Fast"],
    ax=ax,
    annot_kws={"size": 18},
    cbar_kws={"shrink": 0.8}
)
ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
ax.set_title("Cross-Validated Confusion Matrix", fontsize=14, fontweight="bold")

textstr = f"Accuracy: {observed_acc:.1%}\nF1: {cv_f1:.3f}\nAUC: {cv_auc:.3f}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment="center", bbox=props)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix_v2.png", dpi=200, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig_confusion_matrix.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved confusion matrix")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax1 = axes[0]
top_15 = importance_df.head(15)
colors = ["#1f77b4" if "h0" in f else "#ff7f0e" for f in top_15["feature"]]
bars = ax1.barh(range(15), top_15["importance"].values, color=colors, alpha=0.8)
ax1.set_yticks(range(15))
ax1.set_yticklabels(top_15["feature"].values, fontsize=9)
ax1.set_xlabel("Importance")
ax1.set_title("Top 15 Features", fontsize=14, fontweight="bold")
ax1.invert_yaxis()
legend_elements = [
    Patch(facecolor="#1f77b4", alpha=0.8, label="H0 (components)"),
    Patch(facecolor="#ff7f0e", alpha=0.8, label="H1 (cycles)")
]
ax1.legend(handles=legend_elements, loc="lower right")

ax2 = axes[1]
band_sorted = band_importance.sort_values(ascending=True)
band_colors = {
    'delta': '#2196F3', 'theta': '#009688', 'alpha': '#4CAF50',
    'beta': '#FF9800', 'gamma': '#F44336'
}
bar_colors = [band_colors.get(b, '#666666') for b in band_sorted.index]
ax2.barh(band_sorted.index, band_sorted.values, color=bar_colors, alpha=0.85)
ax2.set_xlabel("Total Importance")
ax2.set_title("Feature Importance by Frequency Band", fontsize=14, fontweight="bold")

for i, (band, imp) in enumerate(band_sorted.items()):
    ax2.text(imp + 0.005, i, f"{imp/total_imp*100:.1f}%", va="center", fontsize=11)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance_v2.png", dpi=200, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig_feature_importance.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved feature importance")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax1 = axes[0]
ax1.hist(null_dist, bins=50, alpha=0.7, color="gray", edgecolor="black",
         density=True, label=f"Null distribution (n={N_PERMUTATIONS})")
ax1.axvline(observed_acc, color="red", linewidth=3, linestyle="--",
            label=f"Observed ({observed_acc:.1%})")
ax1.axvline(null_dist.mean(), color="blue", linewidth=2, linestyle=":",
            label=f"Null mean ({null_dist.mean():.1%})")
ax1.axvline(0.5, color="green", linewidth=2, linestyle="-.",
            label="Chance (50%)")
ax1.set_xlabel("Cross-Validation Accuracy", fontweight="bold")
ax1.set_ylabel("Density", fontweight="bold")
ax1.set_title("Permutation Test", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

if p_value < 0.001:
    sig_level = "*** (p < 0.001)"
elif p_value < 0.01:
    sig_level = "** (p < 0.01)"
elif p_value < 0.05:
    sig_level = "* (p < 0.05)"
else:
    sig_level = "ns"

textstr = f"p = {p_value:.4f}\nCohen's d = {effect_size:.2f}\n{sig_level}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment="top", horizontalalignment="right", bbox=props)


ax2 = axes[1]
ax2.hist(boot_accs, bins=50, alpha=0.7, color="steelblue", edgecolor="black",
         density=True, label=f"Bootstrap distribution (n={len(boot_accs)})")
ax2.axvline(observed_acc, color="red", linewidth=3, linestyle="--",
            label=f"Observed ({observed_acc:.1%})")
ax2.axvline(ci_lower, color="orange", linewidth=2, linestyle=":")
ax2.axvline(ci_upper, color="orange", linewidth=2, linestyle=":",
            label=f"95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
ax2.axvspan(ci_lower, ci_upper, alpha=0.2, color="orange")
ax2.axvline(0.5, color="green", linewidth=2, linestyle="-.", label="Chance (50%)")
ax2.set_xlabel("Cross-Validation Accuracy", fontweight="bold")
ax2.set_ylabel("Density", fontweight="bold")
ax2.set_title("Bootstrap 95% Confidence Interval", fontsize=14, fontweight="bold")
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)

textstr = f"95% CI:\n[{ci_lower:.1%}, {ci_upper:.1%}]"
props = dict(boxstyle="round", facecolor="lightblue", alpha=0.9)
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment="top", horizontalalignment="right", bbox=props)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "statistical_tests_v2.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved statistical tests")


# # Move old results
# import shutil
# for old_file in ["results_summary.json", "confusion_matrix.png",
#                  "feature_importance.png", "statistical_tests.png"]:
#     src = RESULTS_DIR / old_file
#     if src.exists():
#         shutil.move(str(src), str(OLD_RESULTS / old_file))
#         print(f"  Archived: {old_file}")


results_dict = {
    "cv_accuracy_mean": float(cv_scores.mean()),
    "cv_accuracy_std": float(cv_scores.std()),
    "cv_scores_per_fold": cv_scores.tolist(),
    "f1_score": float(cv_f1),
    "roc_auc": float(cv_auc),
    "p_value": float(p_value),
    "effect_size_cohens_d": float(effect_size),
    "significance_level": sig_level,
    "ci_lower_bootstrap": float(ci_lower),
    "ci_upper_bootstrap": float(ci_upper),
    "ci_method": "subject-level bootstrap (2000 iterations)",
    "confusion_matrix": cm.tolist(),
    "slow_accuracy_pct": float(slow_correct),
    "fast_accuracy_pct": float(fast_correct),
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "n_subjects": int(len(np.unique(subjects))),
    "n_slow": int(np.sum(y == 0)),
    "n_fast": int(np.sum(y == 1)),
    "model": "RandomForestClassifier",
    "cv_method": cv_name,
    "n_splits": N_SPLITS,
    "n_permutations": N_PERMUTATIONS,
    "band_importance": {
        band: {"importance": float(imp), "pct": float(imp/total_imp*100)}
        for band, imp in band_importance.items()
    },
    "top_features": importance_df.head(20)[["feature", "importance"]].to_dict(orient="records"),
    "conclusion": "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
}

with open(RESULTS_DIR / "results_summary.json", "w") as f:
    json.dump(results_dict, f, indent=2)
print("  Saved results_summary.json")

importance_df.to_csv(RESULTS_DIR / "feature_importance_ranked.csv", index=False)
print("  Saved feature_importance_ranked.csv")

print(f""" resultados
  Accuracy: {observed_acc:.1%} ± {cv_scores.std():.1%} (GroupKFold, k={N_SPLITS})     
  F1 Score: {cv_f1:.3f}                                       
  ROC-AUC:  {cv_auc:.3f}                                       
  p-value:  {p_value:.4f} (permutation, n={N_PERMUTATIONS})             
  Cohen's d: {effect_size:.2f}                                      
  95% CI:   [{ci_lower:.1%}, {ci_upper:.1%}] (subject bootstrap)    
  Slow correct: {slow_correct:.1f}%, Fast correct: {fast_correct:.1f}%        
  Top band: {list(band_importance.items())[0][0]} ({list(band_importance.items())[0][1]/total_imp*100:.1f}%)                                   
"""
)