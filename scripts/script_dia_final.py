# %% [markdown]
# # Phase 3-5: TDA Feature Extraction and Classification (IMPROVED VERSION)
# ## Project 1.b - Connectivity Graph Analysis
#
# **Project Objective**:
# Determine if slow audio induces different EEG connectivity patterns compared to fast audio
# in infants, using Topological Data Analysis on functional connectivity graphs.
#
# **Hypothesis**:
# Slow audio → different connectivity patterns → distinct topological signatures → classifiable
#
# **Pipeline**:
# - **Phase 3**: Extract topological features from distance matrices using Ripser
# - **Phase 4**: Train classifiers with proper subject-level cross-validation
# - **Phase 5**: Statistical validation and hypothesis testing
#
# **Key Requirement**: Subject-level splits to avoid data leakage (multiple recordings per subject)
#
# **Changes from v1**:
# - Fixed section ordering (8 before 9)
# - Added CV-based confusion matrix (not training set)
# - Added persistence diagram visualization
# - Added per-band feature importance analysis
# - Improved edge case handling
# - Added data validation checks
# - Better documentation and interpretation

# %% [markdown]
# ## 1. Setup and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

# TDA libraries
from ripser import ripser
import persim

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    LeaveOneGroupOut,
    GroupKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
np.random.seed(42)

print("=" * 60)
print("TDA EEG Classification Pipeline v2.0")
print("=" * 60)
print("Libraries imported successfully")

# %% [markdown]
# ## 2. Configuration and Data Paths

# %%
# Paths
GRAPHS_DIR = Path("graphs")
FEATURES_DIR = Path("features")
RESULTS_DIR = Path("results")
FEATURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Frequency bands - each captures different neural activity
FREQ_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FREQ_BAND_RANGES = {
    "delta": "0.5-4 Hz (deep sleep, attention)",
    "theta": "4-8 Hz (drowsiness, memory)",
    "alpha": "8-13 Hz (relaxation, inhibition)",
    "beta": "13-30 Hz (alertness, active thinking)",
    "gamma": "30-100 Hz (perception, cognition)",
}

# TDA parameters
MAX_DIM = 1  # Compute H0 (connected components) and H1 (cycles/loops)
# Note: MAX_EDGE_LENGTH should be set based on your distance metric
# If using correlation distance (1 - |corr|): range is [0, 1], use ~1.0
# If using other metrics, adjust accordingly
MAX_EDGE_LENGTH = 2.0  # Maximum filtration value for Rips complex

# Classification parameters
N_SPLITS = 5  # Number of folds for GroupKFold CV
N_PERMUTATIONS = 1000  # For permutation test
N_BOOTSTRAP = 1000  # For confidence intervals
RANDOM_STATE = 42

print("Configuration:")
print(f"  Graph data directory: {GRAPHS_DIR}")
print(f"  Features output: {FEATURES_DIR}")
print(f"  Results output: {RESULTS_DIR}")
print(f"\nFrequency bands:")
for band, desc in FREQ_BAND_RANGES.items():
    print(f"  {band}: {desc}")
print(f"\nTDA parameters:")
print(f"  Max homology dimension: H{MAX_DIM}")
print(f"  Max edge length (filtration): {MAX_EDGE_LENGTH}")
print(f"\nClassification parameters:")
print(f"  CV folds: {N_SPLITS}")
print(f"  Permutation iterations: {N_PERMUTATIONS}")
print(f"  Bootstrap iterations: {N_BOOTSTRAP}")

# %% [markdown]
# # Phase 3: Topological Data Analysis
#
# ## 3.1 Understanding the TDA Approach
#
# **Why TDA for EEG connectivity?**
# - EEG connectivity graphs capture functional relationships between brain regions
# - Traditional graph metrics (density, clustering) lose structural information
# - TDA captures multi-scale topological features that persist across filtration values
#
# **What we compute:**
# - **H0 (connected components)**: How the graph fragments/connects at different thresholds
# - **H1 (cycles/loops)**: Circular patterns in connectivity that might indicate feedback loops
#
# **Hypothesis connection:**
# - Slow audio may induce more synchronized brain activity → different connectivity topology
# - These differences should manifest as different persistence diagrams

# %% [markdown]
# ## 3.2 Persistence Diagram Computation

# %%
def validate_distance_matrix(distance_matrix, name=""):
    """
    Validate that a matrix is a proper distance matrix.
    
    Requirements:
    - Square matrix
    - Symmetric (or nearly symmetric)
    - Non-negative values
    - Zero diagonal
    
    Parameters:
    -----------
    distance_matrix : ndarray
        Matrix to validate
    name : str
        Name for error messages
        
    Returns:
    --------
    is_valid : bool
    issues : list of str
    """
    issues = []
    
    # Check square
    if distance_matrix.ndim != 2:
        issues.append(f"Not 2D: shape={distance_matrix.shape}")
        return False, issues
    
    n, m = distance_matrix.shape
    if n != m:
        issues.append(f"Not square: shape=({n}, {m})")
        return False, issues
    
    # Check symmetry (with tolerance for floating point)
    if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5, atol=1e-8):
        max_diff = np.max(np.abs(distance_matrix - distance_matrix.T))
        issues.append(f"Not symmetric: max asymmetry={max_diff:.6f}")
    
    # Check non-negative
    if np.any(distance_matrix < -1e-10):  # Small tolerance for numerical errors
        min_val = np.min(distance_matrix)
        issues.append(f"Negative values present: min={min_val:.6f}")
    
    # Check diagonal
    diag = np.diag(distance_matrix)
    if not np.allclose(diag, 0, atol=1e-10):
        max_diag = np.max(np.abs(diag))
        issues.append(f"Non-zero diagonal: max={max_diag:.6f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(distance_matrix)):
        issues.append("Contains NaN values")
    if np.any(np.isinf(distance_matrix)):
        issues.append("Contains Inf values")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def compute_persistence_diagram(distance_matrix, max_dim=1, max_edge_length=2.0):
    """
    Compute persistence diagram from distance matrix using Ripser.
    
    The persistence diagram captures topological features (components, cycles)
    that appear and disappear as we increase the distance threshold.
    
    Parameters:
    -----------
    distance_matrix : ndarray, shape (n, n)
        Symmetric distance matrix with zero diagonal
    max_dim : int
        Maximum homology dimension (1 = H0 and H1)
    max_edge_length : float
        Maximum edge length for Rips complex filtration
        
    Returns:
    --------
    diagrams : list of ndarray
        Persistence diagrams for each dimension [H0, H1, ...]
        Each diagram has shape (n_features, 2) with (birth, death) pairs
    """
    # Ensure matrix is symmetric (take average if small differences exist)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Clip any small negative values from numerical errors
    distance_matrix = np.maximum(distance_matrix, 0)
    
    result = ripser(
        distance_matrix,
        maxdim=max_dim,
        thresh=max_edge_length,
        distance_matrix=True
    )
    return result["dgms"]


def extract_persistence_features(diagram, dim_name=""):
    """
    Extract scalar features from a persistence diagram.
    
    Features capture different aspects of the topological structure:
    - Count: How many features exist
    - Birth/Death times: When features appear/disappear
    - Persistence: How long features last (death - birth)
    - Entropy: Distribution of persistence values
    
    Parameters:
    -----------
    diagram : ndarray, shape (n_features, 2)
        Persistence diagram with (birth, death) pairs
    dim_name : str
        Name of dimension for feature naming
        
    Returns:
    --------
    features : dict
        Dictionary of extracted scalar features
    """
    # Remove infinite death times (essential features that never die)
    finite_mask = np.isfinite(diagram).all(axis=1)
    finite_diagram = diagram[finite_mask]
    
    # Count essential features (those with infinite persistence)
    n_essential = np.sum(~finite_mask)
    
    if len(finite_diagram) == 0:
        # No finite features - return zeros
        return {
            "n_features": 0,
            "n_essential": n_essential,
            "mean_birth": 0,
            "std_birth": 0,
            "mean_death": 0,
            "std_death": 0,
            "mean_persistence": 0,
            "std_persistence": 0,
            "max_persistence": 0,
            "total_persistence": 0,
            "persistence_entropy": 0,
        }
    
    births = finite_diagram[:, 0]
    deaths = finite_diagram[:, 1]
    persistence = deaths - births
    
    # Compute persistence entropy (normalized)
    # Higher entropy = more uniform distribution of persistence values
    if len(persistence) > 1 and np.sum(persistence) > 0:
        p_normalized = persistence / np.sum(persistence)
        p_normalized = p_normalized[p_normalized > 0]  # Remove zeros for log
        entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
        # Normalize by max possible entropy
        entropy = entropy / np.log(len(persistence) + 1e-10)
    else:
        entropy = 0
    
    features = {
        "n_features": len(finite_diagram),
        "n_essential": n_essential,
        "mean_birth": np.mean(births),
        "std_birth": np.std(births) if len(births) > 1 else 0,
        "mean_death": np.mean(deaths),
        "std_death": np.std(deaths) if len(deaths) > 1 else 0,
        "mean_persistence": np.mean(persistence),
        "std_persistence": np.std(persistence) if len(persistence) > 1 else 0,
        "max_persistence": np.max(persistence),
        "total_persistence": np.sum(persistence),
        "persistence_entropy": entropy,
    }
    
    return features


# Test on sample data
print("\n" + "=" * 60)
print("Testing TDA Pipeline")
print("=" * 60)

print("\nGenerating sample distance matrix...")
np.random.seed(42)
test_dist = np.random.rand(47, 47)
test_dist = (test_dist + test_dist.T) / 2  # Make symmetric
np.fill_diagonal(test_dist, 0)

# Validate
is_valid, issues = validate_distance_matrix(test_dist, "test")
print(f"Distance matrix validation: {'PASSED' if is_valid else 'FAILED'}")
if issues:
    for issue in issues:
        print(f"  - {issue}")

# Compute persistence
diagrams_test = compute_persistence_diagram(test_dist, MAX_DIM, MAX_EDGE_LENGTH)
print(f"\nPersistence diagrams computed:")
print(f"  H0 (connected components): {len(diagrams_test[0])} features")
print(f"  H1 (cycles/loops): {len(diagrams_test[1])} features")

# Extract features
features_h0 = extract_persistence_features(diagrams_test[0], "H0")
features_h1 = extract_persistence_features(diagrams_test[1], "H1")
print(f"\nFeatures extracted:")
print(f"  H0: {len(features_h0)} scalar features")
print(f"  H1: {len(features_h1)} scalar features")
print(f"  Total per band: {len(features_h0) + len(features_h1)} features")

# %% [markdown]
# ## 3.3 Visualize Persistence Diagrams

# %%
def plot_persistence_diagram(diagrams, title="Persistence Diagram", ax=None):
    """
    Plot persistence diagram with birth vs death times.
    
    Points far from diagonal = persistent features (important)
    Points near diagonal = noise (short-lived features)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['blue', 'orange', 'green']
    labels = ['H0 (components)', 'H1 (cycles)', 'H2']
    
    max_val = 0
    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            finite_mask = np.isfinite(dgm).all(axis=1)
            finite_dgm = dgm[finite_mask]
            
            if len(finite_dgm) > 0:
                ax.scatter(
                    finite_dgm[:, 0], finite_dgm[:, 1],
                    c=colors[dim], label=labels[dim], alpha=0.6, s=50
                )
                max_val = max(max_val, finite_dgm.max())
            
            # Plot essential features (infinite death) at top
            essential = dgm[~finite_mask]
            if len(essential) > 0:
                ax.scatter(
                    essential[:, 0], 
                    [max_val * 1.1] * len(essential),
                    c=colors[dim], marker='^', s=100, alpha=0.8
                )
    
    # Diagonal line (birth = death)
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, label='Diagonal')
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


# Visualize test persistence diagram
fig, ax = plt.subplots(figsize=(8, 8))
plot_persistence_diagram(diagrams_test, "Sample Persistence Diagram (Random Data)", ax)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "sample_persistence_diagram.png", dpi=150)
plt.show()

print("Persistence diagram explanation:")
print("  - Each point represents a topological feature")
print("  - X-axis (birth): distance threshold where feature appears")
print("  - Y-axis (death): distance threshold where feature disappears")
print("  - Distance from diagonal = persistence = importance")
print("  - H0: connected components (how graph fragments)")
print("  - H1: cycles/loops (circular connectivity patterns)")

# %% [markdown]
# ## 4. Process All Files and Extract Features

# %%
def process_file_features(file_dir, freq_bands, max_dim=1, max_edge_length=2.0, verbose=False):
    """
    Process one file: compute TDA features for all windows and bands.
    
    For each frequency band:
    1. Load distance matrices for all time windows
    2. Compute persistence diagram for each window
    3. Extract scalar features from each diagram
    4. Aggregate features across windows (mean/std)
    
    Parameters:
    -----------
    file_dir : Path
        Directory containing distance matrices for one file
    freq_bands : list
        List of frequency band names
    max_dim : int
        Maximum homology dimension
    max_edge_length : float
        Maximum edge length for Rips filtration
    verbose : bool
        Print detailed progress
        
    Returns:
    --------
    features_dict : dict
        Aggregated features for this file
    metadata : dict
        Information about processing (n_windows per band, validation issues)
    """
    file_features = {}
    metadata = {"n_windows": {}, "validation_issues": []}
    
    for band in freq_bands:
        dist_file = file_dir / f"{band}_distances.npy"
        if not dist_file.exists():
            if verbose:
                print(f"  Warning: {band}_distances.npy not found")
            metadata["n_windows"][band] = 0
            continue
        
        # Load distance matrices (n_windows, n_electrodes, n_electrodes)
        try:
            distance_matrices = np.load(dist_file)
        except Exception as e:
            metadata["validation_issues"].append(f"{band}: load error - {e}")
            continue
        
        n_windows = distance_matrices.shape[0]
        metadata["n_windows"][band] = n_windows
        
        if n_windows == 0:
            if verbose:
                print(f"  Warning: {band} has 0 windows")
            continue
        
        # Validate first matrix
        is_valid, issues = validate_distance_matrix(distance_matrices[0], f"{band}[0]")
        if not is_valid:
            metadata["validation_issues"].extend([f"{band}: {i}" for i in issues])
            # Continue anyway - we'll try to fix in compute_persistence_diagram
        
        # Collect features from all windows
        h0_features_list = []
        h1_features_list = []
        
        for i in range(n_windows):
            dist_matrix = distance_matrices[i]
            
            try:
                # Compute persistence diagrams
                diagrams = compute_persistence_diagram(
                    dist_matrix, max_dim, max_edge_length
                )
                
                # Extract features
                h0_feats = extract_persistence_features(diagrams[0], "H0")
                h1_feats = extract_persistence_features(diagrams[1], "H1")
                
                h0_features_list.append(h0_feats)
                h1_features_list.append(h1_feats)
                
            except Exception as e:
                if verbose:
                    print(f"  Error in {band} window {i}: {e}")
                continue
        
        # Check if we got any valid features
        if len(h0_features_list) == 0:
            if verbose:
                print(f"  Warning: No valid windows for {band}")
            continue
        
        # Aggregate across windows (mean and std)
        for feat_name in h0_features_list[0].keys():
            h0_values = [f[feat_name] for f in h0_features_list]
            file_features[f"{band}_h0_{feat_name}_mean"] = np.mean(h0_values)
            file_features[f"{band}_h0_{feat_name}_std"] = np.std(h0_values)
            
            h1_values = [f[feat_name] for f in h1_features_list]
            file_features[f"{band}_h1_{feat_name}_mean"] = np.mean(h1_values)
            file_features[f"{band}_h1_{feat_name}_std"] = np.std(h1_values)
    
    return file_features, metadata


# Test on one file
print("\n" + "=" * 60)
print("Testing Feature Extraction on Real Data")
print("=" * 60)

slow_dirs = list((GRAPHS_DIR / "slow").iterdir())
if len(slow_dirs) > 0:
    test_graph_dir = slow_dirs[0]
    print(f"Testing on: {test_graph_dir.name}")
    
    features_test, metadata_test = process_file_features(
        test_graph_dir, FREQ_BANDS, MAX_DIM, MAX_EDGE_LENGTH, verbose=True
    )
    
    print(f"\nFeature extraction results:")
    print(f"  Total features: {len(features_test)}")
    print(f"  Windows per band: {metadata_test['n_windows']}")
    if metadata_test['validation_issues']:
        print(f"  Validation issues: {metadata_test['validation_issues']}")
    print(f"  Sample features: {list(features_test.keys())[:5]}")
else:
    print("No data found in graphs/slow directory")

# %% [markdown]
# ## 5. Create Full Dataset

# %%
def create_dataset(graphs_dir_slow, graphs_dir_fast, freq_bands, max_dim=1, max_edge_length=2.0):
    """
    Create complete dataset from all files.
    
    Extracts subject IDs from filenames for proper cross-validation.
    Filename format expected: bbXX_utYY (subject_trial)
    
    Returns:
    --------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) - 0=slow, 1=fast
    subjects : ndarray (n_samples,) - subject IDs
    feature_names : list
    filenames : list
    all_metadata : list of dicts
    """
    all_features = []
    all_labels = []
    all_subjects = []
    all_filenames = []
    all_metadata = []
    
    # Process slow files (label = 0)
    slow_dirs = sorted([d for d in graphs_dir_slow.iterdir() if d.is_dir()])
    print(f"\nProcessing {len(slow_dirs)} SLOW audio files...")
    
    for file_dir in tqdm(slow_dirs, desc="Slow"):
        try:
            features, metadata = process_file_features(
                file_dir, freq_bands, max_dim, max_edge_length
            )
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(0)  # slow = 0
                
                # Extract subject ID (format: bbXX_utYY)
                filename = file_dir.name
                parts = filename.split("_")
                subject_id = parts[0] if len(parts) > 0 else filename
                
                all_subjects.append(subject_id)
                all_filenames.append(filename)
                all_metadata.append(metadata)
        except Exception as e:
            print(f"Error processing {file_dir.name}: {e}")
    
    # Process fast files (label = 1)
    fast_dirs = sorted([d for d in graphs_dir_fast.iterdir() if d.is_dir()])
    print(f"Processing {len(fast_dirs)} FAST audio files...")
    
    for file_dir in tqdm(fast_dirs, desc="Fast"):
        try:
            features, metadata = process_file_features(
                file_dir, freq_bands, max_dim, max_edge_length
            )
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(1)  # fast = 1
                
                filename = file_dir.name
                parts = filename.split("_")
                subject_id = parts[0] if len(parts) > 0 else filename
                
                all_subjects.append(subject_id)
                all_filenames.append(filename)
                all_metadata.append(metadata)
        except Exception as e:
            print(f"Error processing {file_dir.name}: {e}")
    
    # Convert to arrays
    df_features = pd.DataFrame(all_features)
    feature_names = list(df_features.columns)
    X = df_features.values
    y = np.array(all_labels)
    subjects = np.array(all_subjects)
    
    print(f"\n{'=' * 60}")
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total samples: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")
    print(f"  Features per band: {X.shape[1] // len(freq_bands)}")
    print(f"\nClass distribution:")
    print(f"  Slow (0): {np.sum(y == 0)} samples ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  Fast (1): {np.sum(y == 1)} samples ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print(f"\nSubject distribution:")
    print(f"  Unique subjects: {len(np.unique(subjects))}")
    
    return X, y, subjects, feature_names, all_filenames, all_metadata


# Create dataset
print("\n" + "=" * 60)
print("Creating Full Dataset")
print("=" * 60)

X, y, subjects, feature_names, filenames, all_metadata = create_dataset(
    GRAPHS_DIR / "slow",
    GRAPHS_DIR / "fast",
    FREQ_BANDS,
    MAX_DIM,
    MAX_EDGE_LENGTH
)

# Save dataset
np.save(FEATURES_DIR / "X.npy", X)
np.save(FEATURES_DIR / "y.npy", y)
np.save(FEATURES_DIR / "subjects.npy", subjects)

with open(FEATURES_DIR / "feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

with open(FEATURES_DIR / "filenames.txt", "w") as f:
    for name in filenames:
        f.write(f"{name}\n")

print(f"\nDataset saved to {FEATURES_DIR}")

# %% [markdown]
# ## 6. Data Preprocessing and Validation

# %%
print("\n" + "=" * 60)
print("Data Preprocessing")
print("=" * 60)

# Check for NaN/Inf values
print("\nChecking for invalid values...")
nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")

# Handle problematic values
nan_mask = np.isnan(X).any(axis=1)
inf_mask = np.isinf(X).any(axis=1)
valid_mask = ~(nan_mask | inf_mask)

n_removed = (~valid_mask).sum()
if n_removed > 0:
    print(f"\nRemoving {n_removed} samples with invalid values...")
    X = X[valid_mask]
    y = y[valid_mask]
    subjects = subjects[valid_mask]
    filenames = [f for f, v in zip(filenames, valid_mask) if v]

print(f"\nCleaned dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Analyze feature distributions
print("\nFeature statistics:")
print(f"  Min value: {X.min():.4f}")
print(f"  Max value: {X.max():.4f}")
print(f"  Mean: {X.mean():.4f}")
print(f"  Std: {X.std():.4f}")

# Check for constant features (zero variance)
feature_stds = X.std(axis=0)
constant_features = feature_stds < 1e-10
n_constant = constant_features.sum()
if n_constant > 0:
    print(f"\nWarning: {n_constant} features have zero variance")
    print("  These will be uninformative for classification")
    constant_names = [feature_names[i] for i in np.where(constant_features)[0]]
    print(f"  Constant features: {constant_names[:5]}...")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized (mean=0, std=1)")

# %% [markdown]
# ## 7. Subject Distribution Analysis

# %%
print("\n" + "=" * 60)
print("Subject Distribution Analysis")
print("=" * 60)

# Create subject-level summary
subject_df = pd.DataFrame({
    "subject": subjects,
    "label": y,
    "label_name": ["slow" if l == 0 else "fast" for l in y]
})

# Samples per subject
subject_counts = subject_df.groupby("subject").size()
print(f"\nSamples per subject:")
print(f"  Mean: {subject_counts.mean():.1f}")
print(f"  Median: {subject_counts.median():.1f}")
print(f"  Min: {subject_counts.min()}")
print(f"  Max: {subject_counts.max()}")

# Label distribution per subject
subject_labels = subject_df.groupby("subject")["label"].agg(["count", "sum", "mean"])
subject_labels.columns = ["total", "n_fast", "prop_fast"]
subject_labels["n_slow"] = subject_labels["total"] - subject_labels["n_fast"]

print(f"\nLabel distribution per subject:")
print(subject_labels.describe())

# Check if subjects have both conditions (within-subject design)
mixed_subjects = subject_labels[(subject_labels["n_slow"] > 0) & (subject_labels["n_fast"] > 0)]
print(f"\nSubjects with both conditions: {len(mixed_subjects)} / {len(subject_labels)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Samples per subject
ax1 = axes[0]
subject_counts.plot(kind="bar", ax=ax1, color="steelblue", alpha=0.7)
ax1.set_xlabel("Subject ID")
ax1.set_ylabel("Number of Samples")
ax1.set_title("Samples per Subject", fontweight="bold")
ax1.axhline(subject_counts.mean(), color="red", linestyle="--", label=f"Mean: {subject_counts.mean():.1f}")
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Label distribution
ax2 = axes[1]
subject_labels[["n_slow", "n_fast"]].plot(kind="bar", stacked=True, ax=ax2, color=["blue", "orange"], alpha=0.7)
ax2.set_xlabel("Subject ID")
ax2.set_ylabel("Number of Samples")
ax2.set_title("Class Distribution per Subject", fontweight="bold")
ax2.legend(["Slow", "Fast"])
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "subject_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# # Phase 4: Classification
#
# ## 8. Cross-Validation Strategy

# %%
print("\n" + "=" * 60)
print("Cross-Validation Strategy")
print("=" * 60)

print(f"""
Why subject-level CV is critical:
---------------------------------
1. Multiple recordings per subject share subject-specific patterns
2. If same subject appears in train and test, model learns subject identity
3. This causes data leakage and inflated accuracy
4. Subject-level CV ensures generalization to NEW subjects

Strategy: GroupKFold (n_splits={N_SPLITS})
- Groups = subjects
- Each fold: ~{100 // N_SPLITS}% of subjects in test set
- No subject appears in both train and test
- More robust than Leave-One-Subject-Out (larger test sets)
""")

# Setup CV
gkf = GroupKFold(n_splits=N_SPLITS)

# Verify CV splits
print("Verifying CV splits...")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=subjects)):
    train_subjects = set(subjects[train_idx])
    test_subjects = set(subjects[test_idx])
    overlap = train_subjects.intersection(test_subjects)
    
    print(f"  Fold {fold + 1}:")
    print(f"    Train: {len(train_idx)} samples, {len(train_subjects)} subjects")
    print(f"    Test: {len(test_idx)} samples, {len(test_subjects)} subjects")
    print(f"    Subject overlap: {len(overlap)} (should be 0)")

# %% [markdown]
# ## 9. Model Training and Evaluation

# %%
print("\n" + "=" * 60)
print("Model Training and Evaluation")
print("=" * 60)

# Initialize model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Cross-validation scores
print(f"\nTraining Random Forest with GroupKFold CV (k={N_SPLITS})...")
cv_scores = cross_val_score(
    rf_model, X_scaled, y, groups=subjects, cv=gkf, scoring="accuracy"
)

print(f"\nCross-validation Results:")
print(f"  Accuracy per fold: {cv_scores}")
print(f"  Mean accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"  Min/Max: {cv_scores.min():.3f} / {cv_scores.max():.3f}")

# Get CV predictions for confusion matrix (IMPORTANT: not training predictions!)
print("\nObtaining cross-validated predictions...")
y_pred_cv = cross_val_predict(rf_model, X_scaled, y, groups=subjects, cv=gkf)

# Calculate additional metrics
cv_f1 = f1_score(y, y_pred_cv, average="weighted")
print(f"  F1 Score (weighted): {cv_f1:.3f}")

# Try to compute AUC if possible
try:
    # Get probability predictions
    y_proba_cv = cross_val_predict(
        rf_model, X_scaled, y, groups=subjects, cv=gkf, method="predict_proba"
    )
    cv_auc = roc_auc_score(y, y_proba_cv[:, 1])
    print(f"  ROC AUC: {cv_auc:.3f}")
except Exception as e:
    print(f"  ROC AUC: Could not compute ({e})")
    cv_auc = None

# Classification report (on CV predictions)
print("\nClassification Report (Cross-Validated):")
print(classification_report(y, y_pred_cv, target_names=["Slow", "Fast"]))

# %% [markdown]
# ## 10. Confusion Matrix (Cross-Validated)

# %%
# IMPORTANT: Using CV predictions, not training predictions!
cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Slow", "Fast"],
    yticklabels=["Slow", "Fast"],
    ax=ax,
    annot_kws={"size": 16}
)
ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
ax.set_title("Confusion Matrix (Cross-Validated Predictions)", fontsize=14, fontweight="bold")

# Add text with metrics
textstr = f"CV Accuracy: {cv_scores.mean():.1%}\nF1 Score: {cv_f1:.3f}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment="center", bbox=props)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix_cv.png", dpi=150)
plt.show()

print("Note: This confusion matrix uses CROSS-VALIDATED predictions,")
print("not training set predictions. This gives an honest estimate of performance.")

# %% [markdown]
# ## 11. Feature Importance Analysis

# %%
print("\n" + "=" * 60)
print("Feature Importance Analysis")
print("=" * 60)

# Train on full dataset for feature importance
rf_model.fit(X_scaled, y)
feature_importance = rf_model.feature_importances_

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
}).sort_values("importance", ascending=False)

# Extract band and dimension info
importance_df["band"] = importance_df["feature"].apply(
    lambda x: x.split("_")[0] if "_" in x else "unknown"
)
importance_df["dimension"] = importance_df["feature"].apply(
    lambda x: "H0" if "_h0_" in x else "H1" if "_h1_" in x else "unknown"
)

# Top 15 features
print("\nTop 15 Most Important Features:")
print("-" * 60)
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Visualize top features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top 15 features
ax1 = axes[0]
top_15 = importance_df.head(15)
colors = ["blue" if "h0" in f else "orange" for f in top_15["feature"]]
ax1.barh(range(15), top_15["importance"].values, color=colors, alpha=0.7)
ax1.set_yticks(range(15))
ax1.set_yticklabels(top_15["feature"].values, fontsize=9)
ax1.set_xlabel("Importance", fontsize=12)
ax1.set_title("Top 15 Most Important Features", fontsize=14, fontweight="bold")
ax1.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="blue", alpha=0.7, label="H0 (components)"),
                   Patch(facecolor="orange", alpha=0.7, label="H1 (cycles)")]
ax1.legend(handles=legend_elements, loc="lower right")

# Plot 2: Importance by frequency band
ax2 = axes[1]
band_importance = importance_df.groupby("band")["importance"].sum().sort_values(ascending=True)
band_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(band_importance)))
ax2.barh(band_importance.index, band_importance.values, color=band_colors)
ax2.set_xlabel("Total Importance", fontsize=12)
ax2.set_title("Feature Importance by Frequency Band", fontsize=14, fontweight="bold")

# Add percentage labels
total_imp = band_importance.sum()
for i, (band, imp) in enumerate(band_importance.items()):
    ax2.text(imp + 0.01, i, f"{imp / total_imp * 100:.1f}%", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
plt.show()

# Summary by band and dimension
print("\nImportance Summary by Band:")
band_summary = importance_df.groupby("band")["importance"].agg(["sum", "mean"]).sort_values("sum", ascending=False)
band_summary["percent"] = band_summary["sum"] / band_summary["sum"].sum() * 100
print(band_summary.round(4))

print("\nImportance Summary by Dimension:")
dim_summary = importance_df.groupby("dimension")["importance"].agg(["sum", "mean"])
dim_summary["percent"] = dim_summary["sum"] / dim_summary["sum"].sum() * 100
print(dim_summary.round(4))

# %% [markdown]
# # Phase 5: Statistical Validation
#
# ## 12. Permutation Test

# %%
print("\n" + "=" * 60)
print("Statistical Validation: Permutation Test")
print("=" * 60)

print("""
Permutation Test:
-----------------
H0 (null hypothesis): Classification accuracy is due to chance
H1 (alternative): Topological features contain discriminative information

Method: Shuffle labels N times, compute CV accuracy each time
P-value: Proportion of permuted accuracies >= observed accuracy
""")

def permutation_test_cv(X, y, groups, model, cv, n_permutations=1000, random_state=42):
    """Permutation test for cross-validation accuracy."""
    # Observed score
    observed_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="accuracy")
    observed_mean = observed_scores.mean()
    
    # Null distribution
    np.random.seed(random_state)
    null_distribution = []
    
    for i in tqdm(range(n_permutations), desc="Permutation test"):
        y_permuted = np.random.permutation(y)
        perm_scores = cross_val_score(model, X, y_permuted, groups=groups, cv=cv, scoring="accuracy")
        null_distribution.append(perm_scores.mean())
    
    null_distribution = np.array(null_distribution)
    
    # P-value (one-tailed, testing if observed > null)
    p_value = (np.sum(null_distribution >= observed_mean) + 1) / (n_permutations + 1)
    
    # Effect size (Cohen's d)
    effect_size = (observed_mean - null_distribution.mean()) / null_distribution.std()
    
    return observed_mean, null_distribution, p_value, effect_size


# Run permutation test
observed_acc, null_dist, p_value, effect_size = permutation_test_cv(
    X_scaled, y, subjects, rf_model, gkf, 
    n_permutations=N_PERMUTATIONS, 
    random_state=RANDOM_STATE
)

print(f"\nPermutation Test Results:")
print(f"  Observed CV accuracy: {observed_acc:.4f} ({observed_acc:.1%})")
print(f"  Null distribution mean: {null_dist.mean():.4f} ({null_dist.mean():.1%})")
print(f"  Null distribution std: {null_dist.std():.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Effect size (Cohen's d): {effect_size:.2f}")

# Interpret effect size
if abs(effect_size) < 0.2:
    effect_interpretation = "negligible"
elif abs(effect_size) < 0.5:
    effect_interpretation = "small"
elif abs(effect_size) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"  Effect interpretation: {effect_interpretation}")

# Interpret significance
alpha = 0.05
if p_value < 0.001:
    sig_level = "*** (p < 0.001)"
elif p_value < 0.01:
    sig_level = "** (p < 0.01)"
elif p_value < 0.05:
    sig_level = "* (p < 0.05)"
else:
    sig_level = "ns (p >= 0.05)"
print(f"  Significance: {sig_level}")

# %% [markdown]
# ## 13. Bootstrap Confidence Interval

# %%
print("\n" + "=" * 60)
print("Statistical Validation: Bootstrap Confidence Interval")
print("=" * 60)

def bootstrap_cv_score(X, y, groups, model, cv, n_bootstrap=1000, random_state=42):
    """Bootstrap confidence interval for CV accuracy via subject resampling."""
    np.random.seed(random_state)
    bootstrap_scores = []
    
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Resample subjects with replacement
        boot_subjects = np.random.choice(unique_subjects, size=n_subjects, replace=True)
        
        # Get indices for selected subjects
        boot_indices = []
        new_groups = []
        for j, subj in enumerate(boot_subjects):
            subj_indices = np.where(groups == subj)[0]
            boot_indices.extend(subj_indices)
            # Assign new group ID to handle duplicate subjects
            new_groups.extend([j] * len(subj_indices))
        
        boot_indices = np.array(boot_indices)
        new_groups = np.array(new_groups)
        
        X_boot = X[boot_indices]
        y_boot = y[boot_indices]
        
        # Check we have both classes and enough groups
        if len(np.unique(y_boot)) < 2:
            continue
        if len(np.unique(new_groups)) < cv.n_splits:
            continue
        
        try:
            boot_scores = cross_val_score(
                model, X_boot, y_boot, groups=new_groups, cv=cv, scoring="accuracy"
            )
            bootstrap_scores.append(boot_scores.mean())
        except Exception:
            continue
    
    return np.array(bootstrap_scores)


# Run bootstrap
bootstrap_accs = bootstrap_cv_score(
    X_scaled, y, subjects, rf_model, gkf,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE
)

# Confidence interval (percentile method)
ci_lower = np.percentile(bootstrap_accs, 2.5)
ci_upper = np.percentile(bootstrap_accs, 97.5)
ci_width = ci_upper - ci_lower

print(f"\nBootstrap Results ({len(bootstrap_accs)} successful iterations):")
print(f"  Mean accuracy: {bootstrap_accs.mean():.4f} ({bootstrap_accs.mean():.1%})")
print(f"  Std: {bootstrap_accs.std():.4f}")
print(f"  Median: {np.median(bootstrap_accs):.4f}")
print(f"\n95% Confidence Interval:")
print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  [{ci_lower:.1%}, {ci_upper:.1%}]")
print(f"  Width: {ci_width:.4f} ({ci_width:.1%})")

# Check if CI excludes chance
if ci_lower > 0.5:
    print(f"\n  ✓ Entire CI is above chance (50%) - robust result")
else:
    print(f"\n  ⚠ CI includes chance level (50%) - result may not be robust")

# %% [markdown]
# ## 14. Visualization of Statistical Tests

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Permutation Test
ax1 = axes[0]
ax1.hist(null_dist, bins=50, alpha=0.7, color="gray", edgecolor="black", density=True,
         label=f"Null distribution (n={N_PERMUTATIONS})")
ax1.axvline(observed_acc, color="red", linewidth=3, linestyle="--",
            label=f"Observed ({observed_acc:.1%})")
ax1.axvline(null_dist.mean(), color="blue", linewidth=2, linestyle=":",
            label=f"Null mean ({null_dist.mean():.1%})")
ax1.axvline(0.5, color="green", linewidth=2, linestyle="-.",
            label="Chance (50%)")

ax1.set_xlabel("Cross-Validation Accuracy", fontsize=12, fontweight="bold")
ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
ax1.set_title("Permutation Test: Null Distribution", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# Add p-value annotation
textstr = f"p = {p_value:.4f}\nCohen's d = {effect_size:.2f}\n{sig_level}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment="top", horizontalalignment="right", bbox=props)

# Plot 2: Bootstrap Distribution
ax2 = axes[1]
ax2.hist(bootstrap_accs, bins=50, alpha=0.7, color="steelblue", edgecolor="black", density=True,
         label=f"Bootstrap distribution (n={len(bootstrap_accs)})")
ax2.axvline(observed_acc, color="red", linewidth=3, linestyle="--",
            label=f"Observed ({observed_acc:.1%})")
ax2.axvline(ci_lower, color="orange", linewidth=2, linestyle=":",
            label=f"95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
ax2.axvline(ci_upper, color="orange", linewidth=2, linestyle=":")
ax2.axvspan(ci_lower, ci_upper, alpha=0.2, color="orange")
ax2.axvline(0.5, color="green", linewidth=2, linestyle="-.", label="Chance (50%)")

ax2.set_xlabel("Cross-Validation Accuracy", fontsize=12, fontweight="bold")
ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
ax2.set_title("Bootstrap: 95% Confidence Interval", fontsize=14, fontweight="bold")
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)

# Add CI annotation
textstr = f"95% CI:\n[{ci_lower:.1%}, {ci_upper:.1%}]\nWidth: {ci_width:.1%}"
props = dict(boxstyle="round", facecolor="lightblue", alpha=0.9)
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment="top", horizontalalignment="right", bbox=props)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "statistical_tests.png", dpi=150)
plt.show()

# %% [markdown]
# ## 15. Final Results and Conclusions

# %%
print("\n" + "=" * 70)
print("FINAL RESULTS AND CONCLUSIONS")
print("=" * 70)

# Summary table
print("\n" + "-" * 70)
print("SUMMARY TABLE")
print("-" * 70)

results_summary = {
    "Metric": [
        "Dataset size",
        "Number of features",
        "Number of subjects",
        "Class balance (Slow/Fast)",
        "",
        "CV Accuracy (GroupKFold-5)",
        "F1 Score (weighted)",
        "Baseline (chance)",
        "Improvement over baseline",
        "",
        "P-value (permutation test)",
        "Effect size (Cohen's d)",
        "95% CI lower bound",
        "95% CI upper bound",
        "CI above chance?",
    ],
    "Value": [
        f"{X.shape[0]} samples",
        f"{X.shape[1]} features",
        f"{len(np.unique(subjects))} subjects",
        f"{np.sum(y == 0)} / {np.sum(y == 1)}",
        "",
        f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}",
        f"{cv_f1:.3f}",
        "50%",
        f"+{(cv_scores.mean() - 0.5) * 100:.1f} percentage points",
        "",
        f"{p_value:.6f} {sig_level}",
        f"{effect_size:.2f} ({effect_interpretation})",
        f"{ci_lower:.1%}",
        f"{ci_upper:.1%}",
        "Yes" if ci_lower > 0.5 else "No",
    ],
}

df_results = pd.DataFrame(results_summary)
print(df_results.to_string(index=False))

# Most important bands
print("\n" + "-" * 70)
print("MOST DISCRIMINATIVE FREQUENCY BANDS")
print("-" * 70)
print(band_summary.round(3).to_string())

# Interpretation
print("\n" + "-" * 70)
print("INTERPRETATION")
print("-" * 70)

print(f"""
Research Question:
  Can we distinguish between slow and fast audio conditions in infants
  based on EEG connectivity topology?

Results:
  - Cross-validated accuracy: {cv_scores.mean():.1%} (chance = 50%)
  - This is {(cv_scores.mean() - 0.5) * 100:.1f} percentage points above chance
  - P-value = {p_value:.6f} → {"statistically significant" if p_value < 0.05 else "not statistically significant"}
  - Effect size = {effect_size:.2f} → {effect_interpretation} practical significance
  - 95% CI = [{ci_lower:.1%}, {ci_upper:.1%}]
""")

# Evidence level
if cv_scores.mean() > 0.65 and p_value < 0.05 and ci_lower > 0.5:
    evidence = "STRONG"
    conclusion = """
The topological features of EEG connectivity graphs successfully distinguish
between slow and fast audio conditions. This suggests that:

1. Slow vs fast audio induces measurably different brain connectivity patterns
2. These differences are robust across subjects (not subject-specific artifacts)
3. Topological Data Analysis captures meaningful neural signal differences

The most discriminative frequency bands provide insight into which types of
neural oscillations are most affected by audio speed."""

elif cv_scores.mean() > 0.55 and p_value < 0.05:
    evidence = "MODERATE"
    conclusion = """
There is moderate evidence that topological features can distinguish conditions.
The accuracy is above chance and statistically significant, but the effect
size is modest. This suggests:

1. Some differences in connectivity topology exist between conditions
2. The signal may be weak or variable across subjects
3. Consider: more subjects, different features, or alternative approaches"""

else:
    evidence = "WEAK/NONE"
    conclusion = """
The topological features do not reliably distinguish between conditions.
Possible explanations:

1. Audio speed may not significantly affect EEG connectivity topology
2. The effect exists but is too subtle for current methods to detect
3. More subjects or different preprocessing may be needed
4. Alternative TDA approaches (e.g., different filtrations) could be explored"""

print(f"Evidence Level: {evidence}")
print(conclusion)

# Save results
results_dict = {
    "cv_accuracy_mean": cv_scores.mean(),
    "cv_accuracy_std": cv_scores.std(),
    "cv_f1_score": cv_f1,
    "p_value": p_value,
    "effect_size": effect_size,
    "ci_lower": ci_lower,
    "ci_upper": ci_upper,
    "n_samples": X.shape[0],
    "n_features": X.shape[1],
    "n_subjects": len(np.unique(subjects)),
    "evidence_level": evidence,
}

# Save as JSON
import json
with open(RESULTS_DIR / "results_summary.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}/")
print("  - confusion_matrix_cv.png")
print("  - feature_importance.png")
print("  - statistical_tests.png")
print("  - subject_distribution.png")
print("  - sample_persistence_diagram.png")
print("  - results_summary.json")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)