
# Proyecto 1.b - Análisis del gráfico de conectividad funcional


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
import json
from ripser import ripser
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,GroupKFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
np.random.seed(42)




### Configuración y rutas de datos

GRAPHS_DIR = Path("graphs")
FEATURES_DIR = Path("features")
RESULTS_DIR = Path("results")
FEATURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Bandas de frecuencia 
FREQ_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
FREQ_BAND_RANGES = {
    "delta": "0.5-4 Hz (sueño profundo, atención)",
    "theta": "4-8 Hz (somnolencia, memoria)",
    "alpha": "8-13 Hz (relajación, inhibición)",
    "beta": "13-30 Hz (alerta, pensamiento activo)",
    "gamma": "30-100 Hz (percepción, cognición)",
}

# Parámetros 
MAX_DIM = 1  
MAX_EDGE_LENGTH = 2
N_SPLITS = 5 
N_PERMUTATIONS = 1000 
N_BOOTSTRAP = 1000 
RANDOM_STATE = 42

print("Configuración:")
print(f"  Directorio de grafos: {GRAPHS_DIR}")
print(f"  Salida de características: {FEATURES_DIR}")
print(f"  Salida de resultados: {RESULTS_DIR}")
print("Bandas de frecuencia:")
for band, desc in FREQ_BAND_RANGES.items():
    print(f"  {band}: {desc}")
print("Parámetros TDA:")
print(f"  Dimensión de homología máxima: H{MAX_DIM}")
print(f"  Longitud máxima de arista (filtración): {MAX_EDGE_LENGTH}")
print("\nParámetros de clasificación:")
print(f"  Folds de CV: {N_SPLITS}")
print(f"  Iteraciones de permutación: {N_PERMUTATIONS}")
print(f"  Iteraciones de bootstrap: {N_BOOTSTRAP}")


# tda

## diagrama de persistencia


def validate_distance_matrix(distance_matrix, name=""):
    issues = []
    
    if distance_matrix.ndim != 2:
        issues.append(f"No es 2D: forma={distance_matrix.shape}")
        return False, issues
    
    n, m = distance_matrix.shape
    if n != m:
        issues.append(f"No es cuadrada: forma=({n}, {m})")
        return False, issues
    
    if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5, atol=1e-8):
        max_diff = np.max(np.abs(distance_matrix - distance_matrix.T))
        issues.append(f"No simétrica: asimetría máxima={max_diff:.6f}")
    
    if np.any(distance_matrix < -1e-10):
        min_val = np.min(distance_matrix)
        issues.append(f"Valores negativos presentes: min={min_val:.6f}")
    
    diag = np.diag(distance_matrix)
    if not np.allclose(diag, 0, atol=1e-10):
        max_diag = np.max(np.abs(diag))
        issues.append(f"Diagonal no cero: max={max_diag:.6f}")
    
    if np.any(np.isnan(distance_matrix)):
        issues.append("Contiene valores NaN")
    if np.any(np.isinf(distance_matrix)):
        issues.append("Contiene valores Inf")
    is_valid = len(issues) == 0
    return is_valid, issues


def compute_persistence_diagram(distance_matrix, max_dim=1, max_edge_length=2.0):
    """
    Calcular diagrama de persistencia desde matriz de distancia usando Ripser.
    
    El diagrama de persistencia captura características topológicas (componentes, ciclos)
    que aparecen y desaparecen al aumentar el umbral de distancia.
    
    Parámetros:
    -----------
    distance_matrix : ndarray, forma (n, n)
        Matriz de distancia simétrica con diagonal cero
    max_dim : int
        Dimensión de homología máxima (1 = H0 y H1)
    max_edge_length : float
        Longitud máxima de arista para filtración del complejo de Rips
        
    Retorna:
    --------
    diagrams : list of ndarray
        Diagramas de persistencia para cada dimensión [H0, H1, ...]
        Cada diagrama tiene forma (n_features, 2) con pares (nacimiento, muerte)
    """
    # Asegurar que la matriz es simétrica (promediar si existen pequeñas diferencias)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Recortar cualquier valor negativo pequeño por errores numéricos
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
    Extraer características escalares desde un diagrama de persistencia.
    
    Las características capturan diferentes aspectos de la estructura topológica:
    - Conteo: Cuántas características existen
    - Tiempos de nacimiento/muerte: Cuándo aparecen/desaparecen las características
    - Persistencia: Cuánto duran las características (muerte - nacimiento)
    - Entropía: Distribución de valores de persistencia
    
    Parámetros:
    -----------
    diagram : ndarray, forma (n_features, 2)
        Diagrama de persistencia con pares (nacimiento, muerte)
    dim_name : str
        Nombre de dimensión para nombrar características
        
    Retorna:
    --------
    features : dict
        Diccionario de características escalares extraídas
    """
    # Remover tiempos de muerte infinitos (características esenciales que nunca mueren)
    finite_mask = np.isfinite(diagram).all(axis=1)
    finite_diagram = diagram[finite_mask]
    
    # Contar características esenciales (aquellas con persistencia infinita)
    n_essential = np.sum(~finite_mask)
    
    if len(finite_diagram) == 0:
        # No hay características finitas - retornar ceros
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
  
    if len(persistence) > 1 and np.sum(persistence) > 0:
        p_normalized = persistence / np.sum(persistence)
        p_normalized = p_normalized[p_normalized > 0]  
        entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
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


# Probar con datos de muestra
print("Prueba del Pipeline ")
np.random.seed(42)
test_dist = np.random.rand(47, 47)
test_dist = (test_dist + test_dist.T) / 2  
np.fill_diagonal(test_dist, 0)


is_valid, issues = validate_distance_matrix(test_dist, "test")
print(f"Validación de matriz de distancia: {'APROBADA' if is_valid else 'FALLIDA'}")
if issues:
    for issue in issues:
        print(f"  - {issue}")

# Calcular persistencia
diagrams_test = compute_persistence_diagram(test_dist, MAX_DIM, MAX_EDGE_LENGTH)
print("Diagramas de persistencia calculados:")
print(f"  H0 (componentes conectadas): {len(diagrams_test[0])} características")
print(f"  H1 (ciclos/loops): {len(diagrams_test[1])} características")

# Extraer características
features_h0 = extract_persistence_features(diagrams_test[0], "H0")
features_h1 = extract_persistence_features(diagrams_test[1], "H1")
print("Características extraídas:")
print(f"  H0: {len(features_h0)} características escalares")
print(f"  H1: {len(features_h1)} características escalares")
print(f"  Total por banda: {len(features_h0) + len(features_h1)} características")



def plot_persistence_diagram(diagrams, title="Diagrama de Persistencia", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['blue', 'orange', 'green']
    labels = ['H0 (componentes)', 'H1 (ciclos)', 'H2']
    
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
            
            essential = dgm[~finite_mask]
            if len(essential) > 0:
                ax.scatter(
                    essential[:, 0], 
                    [max_val * 1.1] * len(essential),
                    c=colors[dim], marker='^', s=100, alpha=0.8
                )
    
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, label='Diagonal')
    ax.set_xlabel('Nacimiento', fontsize=12)
    ax.set_ylabel('Muerte', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


fig, ax = plt.subplots(figsize=(8, 8))
plot_persistence_diagram(diagrams_test, "Diagrama de Persistencia de Muestra (Datos Aleatorios)", ax)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "sample_persistence_diagram.png", dpi=150)
plt.show()

print("Explicación del diagrama de persistencia:")
print("  - Cada punto representa una característica topológica")
print("  - Eje X (nacimiento): umbral de distancia donde aparece la característica")
print("  - Eje Y (muerte): umbral de distancia donde desaparece la característica")
print("  - Distancia desde diagonal = persistencia = importancia")
print("  - H0: componentes conectadas (cómo se fragmenta el grafo)")
print("  - H1: ciclos/loops (patrones de conectividad circular)")

## Procesar todos los archivos y extraer características


def process_file_features(file_dir, freq_bands, max_dim=1, max_edge_length=2.0, verbose=False):
    file_features = {}
    metadata = {"n_windows": {}, "validation_issues": []}
    
    for band in freq_bands:
        dist_file = file_dir / f"{band}_distances.npy"
        if not dist_file.exists():
            if verbose:
                print(f"  Warning: {band}_distances.npy not found")
            metadata["n_windows"][band] = 0
            continue
        
        try:
            distance_matrices = np.load(dist_file)
        except Exception as e:
            metadata["validation_issues"].append(f"{band}: error de carga - {e}")
            continue
        
        n_windows = distance_matrices.shape[0]
        metadata["n_windows"][band] = n_windows
        
        if n_windows == 0:
            if verbose:
                print(f"  Warning: {band} has 0 windows")
            continue
        

        max_validation_windows = 10
        n_validation = min(n_windows, max_validation_windows)
        for window_idx in range(n_validation):
            is_valid, issues = validate_distance_matrix(
                distance_matrices[window_idx], f"{band}[{window_idx}]"
            )
            if not is_valid:
                metadata["validation_issues"].extend(
                    [f"{band}[{window_idx}]: {issue}" for issue in issues]
                )

        h0_features_list = []
        h1_features_list = []
        
        for i in range(n_windows):
            dist_matrix = distance_matrices[i]
            
            try:
                # Calcular diagramas de persistencia
                diagrams = compute_persistence_diagram(
                    dist_matrix, max_dim, max_edge_length
                )
                
                # Extraer características
                h0_feats = extract_persistence_features(diagrams[0], "H0")
                h1_feats = extract_persistence_features(diagrams[1], "H1")
                
                h0_features_list.append(h0_feats)
                h1_features_list.append(h1_feats)
                
            except Exception as e:
                if verbose:
                    print(f"  Error in {band} window {i}: {e}")
                continue
        
        # Verificar si obtuvimos características válidas
        if len(h0_features_list) == 0:
            if verbose:
                print(f"  Advertencia: No hay ventanas válidas para {band}")
            continue
        
        # Agregar a través de ventanas (media y std)
        for feat_name in h0_features_list[0].keys():
            h0_values = [f[feat_name] for f in h0_features_list]
            file_features[f"{band}_h0_{feat_name}_mean"] = np.mean(h0_values)
            file_features[f"{band}_h0_{feat_name}_std"] = np.std(h0_values)
            
            h1_values = [f[feat_name] for f in h1_features_list]
            file_features[f"{band}_h1_{feat_name}_mean"] = np.mean(h1_values)
            file_features[f"{band}_h1_{feat_name}_std"] = np.std(h1_values)
    
    return file_features, metadata



print("Prueba de Extracción de Características en Datos Reales")


slow_dirs = list((GRAPHS_DIR / "slow").iterdir())
if len(slow_dirs) > 0:
    test_graph_dir = slow_dirs[0]
    print(f"Probando en: {test_graph_dir.name}")
    
    features_test, metadata_test = process_file_features(
        test_graph_dir, FREQ_BANDS, MAX_DIM, MAX_EDGE_LENGTH, verbose=True
    )
    
    print("Resultados de extracción de características:")
    print(f"  Total de características: {len(features_test)}")
    print(f"  Ventanas por banda: {metadata_test['n_windows']}")
    if metadata_test['validation_issues']:
        print(f"  Problemas de validación: {metadata_test['validation_issues']}")
    print(f"  Características de muestra: {list(features_test.keys())[:5]}")
else:
    print("No se encontraron datos en el directorio graphs/slow")


def create_dataset(graphs_dir_slow, graphs_dir_fast, freq_bands, max_dim=1, max_edge_length=2.0):
    all_features = []
    all_labels = []
    all_subjects = []
    all_filenames = []
    all_metadata = []
    

    slow_dirs = sorted([d for d in graphs_dir_slow.iterdir() if d.is_dir()])
    print(f"\nProcesando {len(slow_dirs)} archivos de audio LENTO...")
    
    for file_dir in tqdm(slow_dirs, desc="Lento"):
        try:
            features, metadata = process_file_features(
                file_dir, freq_bands, max_dim, max_edge_length
            )
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(0)  
                filename = file_dir.name
                parts = filename.split("_")
                subject_id = parts[0] if len(parts) > 0 else filename
                
                all_subjects.append(subject_id)
                all_filenames.append(filename)
                all_metadata.append(metadata)
        except Exception as e:
            print(f"Error procesando {file_dir.name}: {e}")
    

    fast_dirs = sorted([d for d in graphs_dir_fast.iterdir() if d.is_dir()])
    print(f"Procesando {len(fast_dirs)} archivos de audio RÁPIDO...")
    
    for file_dir in tqdm(fast_dirs, desc="Rápido"):
        try:
            features, metadata = process_file_features(
                file_dir, freq_bands, max_dim, max_edge_length
            )
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(1)  # rápido = 1
                
                filename = file_dir.name
                parts = filename.split("_")
                subject_id = parts[0] if len(parts) > 0 else filename
                
                all_subjects.append(subject_id)
                all_filenames.append(filename)
                all_metadata.append(metadata)
        except Exception as e:
            print(f"Error procesando {file_dir.name}: {e}")
    

    df_features = pd.DataFrame(all_features)
    feature_names = list(df_features.columns)
    X = df_features.values
    y = np.array(all_labels)
    subjects = np.array(all_subjects)
    
    print("Resumen del Dataset")
    print("-" * 60)
    print(f"Total de muestras: {X.shape[0]}")
    print(f"Total de características: {X.shape[1]}")
    print(f"  Características por banda: {X.shape[1] // len(freq_bands)}")
    print("\nDistribución de clases:")
    print(f"  Lento (0): {np.sum(y == 0)} muestras ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  Rápido (1): {np.sum(y == 1)} muestras ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    print("\nDistribución de sujetos:")
    print(f"  Sujetos únicos: {len(np.unique(subjects))}")
    
    return X, y, subjects, feature_names, all_filenames, all_metadata

X, y, subjects, feature_names, filenames, all_metadata = create_dataset(
    GRAPHS_DIR / "slow",
    GRAPHS_DIR / "fast",
    FREQ_BANDS,
    MAX_DIM,
    MAX_EDGE_LENGTH
)

# Guardar dataset
np.save(FEATURES_DIR / "X.npy", X)
np.save(FEATURES_DIR / "y.npy", y)
np.save(FEATURES_DIR / "subjects.npy", subjects)

with open(FEATURES_DIR / "feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

with open(FEATURES_DIR / "filenames.txt", "w") as f:
    for name in filenames:
        f.write(f"{name}\n")

print(f"\nDataset guardado en {FEATURES_DIR}")



print("Preprocesamiento de Datos")



nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
print(f"  Valores NaN: {nan_count}")
print(f"  Valores Inf: {inf_count}")

nan_mask = np.isnan(X).any(axis=1)
inf_mask = np.isinf(X).any(axis=1)
valid_mask = ~(nan_mask | inf_mask)

n_removed = (~valid_mask).sum()
if n_removed > 0:
    print(f"\nEliminando {n_removed} muestras con valores inválidos")
    X = X[valid_mask]
    y = y[valid_mask]
    subjects = subjects[valid_mask]
    filenames = [f for f, v in zip(filenames, valid_mask) if v]

print(f"\nDataset limpio: {X.shape[0]} muestras, {X.shape[1]} características")

print("\nEstadísticas de características:")
print(f"  Valor mínimo: {X.min():.4f}")
print(f"  Valor máximo: {X.max():.4f}")
print(f"  Media: {X.mean():.4f}")
print(f"  Desviación estándar: {X.std():.4f}")

feature_stds = X.std(axis=0)
constant_features = feature_stds < 1e-10
n_constant = constant_features.sum()
if n_constant > 0:
    print(f"\nAdvertencia: {n_constant} características tienen varianza cero")
    print("  Estas no serán informativas para la clasificación")
    constant_names = [feature_names[i] for i in np.where(constant_features)[0]]
    print(f" Características constantes: {constant_names[:5]}...")


print("Análisis de Distribución de Sujetos")

subject_df = pd.DataFrame({
    "subject": subjects,
    "label": y,
    "label_name": ["lento" if label == 0 else "rapido" for label in y]
})

subject_counts = subject_df.groupby("subject").size()
print("\nMuestras por sujeto:")
print(f"  Media: {subject_counts.mean():.1f}")
print(f"  Mediana: {subject_counts.median():.1f}")
print(f"  Mínimo: {subject_counts.min()}")
print(f"  Máximo: {subject_counts.max()}")

subject_labels = subject_df.groupby("subject")["label"].agg(["count", "sum", "mean"])
subject_labels.columns = ["total", "n_rapido", "prop_rapido"]
subject_labels["n_lento"] = subject_labels["total"] - subject_labels["n_rapido"]

print("\nDistribución de etiquetas por sujeto:")
print(subject_labels.describe())

mixed_subjects = subject_labels[(subject_labels["n_lento"] > 0) & (subject_labels["n_rapido"] > 0)]
print(f"\nSujetos con ambas condiciones: {len(mixed_subjects)} / {len(subject_labels)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
subject_counts.plot(kind="bar", ax=ax1, color="steelblue", alpha=0.7)
ax1.set_xlabel("ID de Sujeto")
ax1.set_ylabel("Número de Muestras")
ax1.set_title("Muestras por Sujeto", fontweight="bold")
ax1.axhline(subject_counts.mean(), color="red", linestyle="--", label=f"Media: {subject_counts.mean():.1f}")
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

ax2 = axes[1]
subject_labels[["n_lento", "n_rapido"]].plot(kind="bar", stacked=True, ax=ax2, color=["blue", "orange"], alpha=0.7)
ax2.set_xlabel("ID de Sujeto")
ax2.set_ylabel("Número de Muestras")
ax2.set_title("Distribución de Clases por Sujeto", fontweight="bold")
ax2.legend(["Lento", "Rápido"])
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "subject_distribution.png", dpi=150)
plt.show()




gkf = GroupKFold(n_splits=N_SPLITS)


for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects)):
    train_subjects = set(subjects[train_idx])
    test_subjects = set(subjects[test_idx])
    overlap = train_subjects.intersection(test_subjects)
    
    print(f"Fold {fold + 1}:")
    print(f"Entrenamiento: {len(train_idx)} muestras, {len(train_subjects)} sujetos")
    print(f"Prueba: {len(test_idx)} muestras, {len(test_subjects)} sujetos")
    print(f"Solapamiento de sujetos: {len(overlap)} (debería ser 0)")
    
    if len(overlap) > 0:
        print(f"Advertencia: Solapamiento de sujetos detectado: {overlap}")
    else:
        print("    Sin solapamiento de sujetos - aislamiento apropiado")


print("Entrenamiento y Evaluación del Modelo")

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


cv_scores = cross_val_score(
    pipeline, X, y, groups=subjects, cv=gkf, scoring="accuracy"
)

print("\nResultados de validación cruzada:")
print(f"  Precisión por fold: {cv_scores}")
print(f"  Precisión media: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"  Mín/Máx: {cv_scores.min():.3f} / {cv_scores.max():.3f}")

y_pred_cv = cross_val_predict(pipeline, X, y, groups=subjects, cv=gkf)
cv_f1 = f1_score(y, y_pred_cv, average="weighted")
print(f"  Puntaje F1 (ponderado): {cv_f1:.3f}")

y_proba_cv = cross_val_predict(pipeline, X, y, groups=subjects, cv=gkf, method="predict_proba")
cv_auc = roc_auc_score(y, y_proba_cv[:, 1])
print(f"ROC AUC: {cv_auc:.3f}")

report=classification_report(y, y_pred_cv, target_names=["Lento", "Rápido"])
print(report)
report_df = pd.DataFrame(report.split('\n'))
report_df.to_csv( "Reporte_resultados.csv", index=False)

cm = confusion_matrix(y, y_pred_cv)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Lento", "Rápido"],
    yticklabels=["Lento", "Rápido"],
    ax=ax,
    annot_kws={"size": 16}
)
ax.set_xlabel("Predicción", fontsize=12, fontweight="bold")
ax.set_ylabel("Real", fontsize=12, fontweight="bold")
ax.set_title("Matriz de Confusión (Cross Validation)", fontsize=14, fontweight="bold")


textstr = f"Precisión de CV: {cv_scores.mean():.1%}\nPuntaje F1: {cv_f1:.3f}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment="center", bbox=props)

plt.tight_layout()
plt.savefig("Matriz_confusion_prueba.png", dpi=150)
plt.show()



print("Análisis de Importancia de Características")

pipeline.fit(X, y)
feature_importance = pipeline.named_steps['classifier'].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance
}).sort_values("importance", ascending=False)

importance_df["band"] = importance_df["feature"].apply(
    lambda x: x.split("_")[0] if "_" in x else "unknown"
)
importance_df["dimension"] = importance_df["feature"].apply(
    lambda x: "H0" if "_h0_" in x else "H1" if "_h1_" in x else "unknown"
)

print("\nCaracterísticas Más Importantes:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
top_15 = importance_df.head(15)
colors = ["blue" if "h0" in f else "orange" for f in top_15["feature"]]
ax1.barh(range(15), top_15["importance"].values, color=colors, alpha=0.7)
ax1.set_yticks(range(15))
ax1.set_yticklabels(top_15["feature"].values, fontsize=9)
ax1.set_xlabel("Importancia", fontsize=12)
ax1.set_title("15 Características Más Importantes", fontsize=14, fontweight="bold")
ax1.invert_yaxis()


legend_elements = [Patch(facecolor="blue", alpha=0.7, label="H0 (componentes)"),
                   Patch(facecolor="orange", alpha=0.7, label="H1 (ciclos)")]
ax1.legend(handles=legend_elements, loc="lower right")

ax2 = axes[1]
band_importance = importance_df.groupby("band")["importance"].sum().sort_values(ascending=True)
band_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(band_importance)))
ax2.barh(band_importance.index, band_importance.values, color=band_colors)
ax2.set_xlabel("Importancia total", fontsize=12)
ax2.set_title("Importancia de las Características por Banda de Frecuencia", fontsize=14, fontweight="bold")

total_imp = band_importance.sum()
for i, (band, imp) in enumerate(band_importance.items()):
    ax2.text(imp + 0.01, i, f"{imp / total_imp * 100:.1f}%", va="center", fontsize=10)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
plt.show()


print("\nResumen por Banda:")
band_summary = importance_df.groupby("band")["importance"].agg(["sum", "mean"]).sort_values("sum", ascending=False)
band_summary["porcentaje"] = band_summary["sum"] / band_summary["sum"].sum() * 100
print(band_summary.round(4))

print("\nResumen por Dimensión:")
dim_summary = importance_df.groupby("dimension")["importance"].agg(["sum", "mean"])
dim_summary["porcentaje"] = dim_summary["sum"] / dim_summary["sum"].sum() * 100
print(dim_summary.round(4))


# Validación estadística

def permutation_test_cv(X, y, subjects, model, cv, n_permutations=1000, random_state=42):
    observed_acc = cross_val_score(model, X, y, groups=subjects, cv=cv, scoring="accuracy").mean()
    
    rng = np.random.RandomState(random_state)
    null_dist = []
    
    for i in tqdm(range(n_permutations), desc="Test de permutación"):
        y_perm = rng.permutation(y)
        perm_acc = cross_val_score(model, X, y_perm, groups=subjects, cv=cv, scoring="accuracy").mean()
        null_dist.append(perm_acc)
    
    null_dist = np.array(null_dist)
    
    p_value = (np.sum(null_dist >= observed_acc) + 1) / (n_permutations + 1)
    
    effect_size = (observed_acc - null_dist.mean()) / null_dist.std()
    
    return observed_acc, null_dist, p_value, effect_size


# Test de permutación

observed_acc, null_dist, p_value, effect_size = permutation_test_cv(
    X, y, subjects, pipeline, gkf, 
    n_permutations=N_PERMUTATIONS, 
    random_state=RANDOM_STATE)

print(f"  Precisión de CV observada: {observed_acc:.4f} ({observed_acc:.1%})")
print(f"  Media de distribución nula: {null_dist.mean():.4f} ({null_dist.mean():.1%})")
print(f"  Desviación estándar de distribución nula: {null_dist.std():.4f}")
print(f"  Valor p: {p_value:.6f}")
print(f"  Tamaño de efecto (Cohen's d): {effect_size:.2f}")

if abs(effect_size) < 0.2:
    effect_interpretation = "despreciable"
elif abs(effect_size) < 0.5:
    effect_interpretation = "pequeña"
elif abs(effect_size) < 0.8:
    effect_interpretation = "mediana"
else:
    effect_interpretation = "grande"
print(f"  Interpretación de efecto: {effect_interpretation}")

alpha = 0.05
if p_value < 0.001:
    sig_level = "*** (p < 0.001)"
elif p_value < 0.01:
    sig_level = "** (p < 0.01)"
elif p_value < 0.05:
    sig_level = "* (p < 0.05)"
else:
    sig_level = "ns (p >= 0.05)"
print(f"  Significancia: {sig_level}")


# Intervalo de confianza bootstrap

def bootstrap_cv_score(X, y, groups, model, cv, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)
    bootstrap_scores = []
    
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        boot_subjects = np.random.choice(unique_subjects, size=n_subjects, replace=True)
        
        boot_indices = []
        new_groups = []
        for j, subj in enumerate(boot_subjects):
            subj_indices = np.where(groups == subj)[0]
            boot_indices.extend(subj_indices)
            new_groups.extend([j] * len(subj_indices))
        
        boot_indices = np.array(boot_indices)
        new_groups = np.array(new_groups)
        
        X_boot = X[boot_indices]
        y_boot = y[boot_indices]
        
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

bootstrap_accs = bootstrap_cv_score(
    X, y, subjects, pipeline, gkf,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE
)

ci_lower = np.percentile(bootstrap_accs, 2.5)
ci_upper = np.percentile(bootstrap_accs, 97.5)
ci_width = ci_upper - ci_lower

print(f"\nBootstrap Results ({len(bootstrap_accs)} iteraciones exitosas):")
print(f"  Media de precisión: {bootstrap_accs.mean():.4f} ({bootstrap_accs.mean():.1%})")
print(f"  Desviación estándar: {bootstrap_accs.std():.4f}")
print(f"  Mediana: {np.median(bootstrap_accs):.4f}")
print("95% Intervalo de confianza:")
print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"  [{ci_lower:.1%}, {ci_upper:.1%}]")
print(f"  Ancho: {ci_width:.4f} ({ci_width:.1%})")

if ci_lower > 0.5:
    
    print("El IC completo está por encima del nivel del azar (50%) - resultado robusto")
else:
    print("El IC incluye el nivel del azar (50%) - el resultado podría no ser robusto")


### Visualización de los tests estadísticos


fig, axes = plt.subplots(1, 2, figsize=(16, 6))


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


textstr = f"p = {p_value:.4f}\nCohen's d = {effect_size:.2f}\n{sig_level}"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.9)
ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment="top", horizontalalignment="right", bbox=props)

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

textstr = f"95% CI:\n[{ci_lower:.1%}, {ci_upper:.1%}]\nWidth: {ci_width:.1%}"
props = dict(boxstyle="round", facecolor="lightblue", alpha=0.9)
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment="top", horizontalalignment="right", bbox=props)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "statistical_tests.png", dpi=150)
plt.show()


results_summary = {
    "Métrica": [
        "Tamaño del dataset",
        "Número de características",
        "Número de sujetos",
        "Balance de clases (Lento/Rápido)",
        "",
        "Precisión CV (GroupKFold-5)",
        "Puntaje F1 (ponderado)",
        "Línea base (azar)",
        "Mejora sobre línea base",
        "",
        "Valor-p (test de permutación)",
        "Tamaño del efecto (d de Cohen)",
        "Límite inferior IC 95%",
        "Límite superior IC 95%",
        "IC sobre azar?",
    ],
    "Valor": [
        f"{X.shape[0]} muestras",
        f"{X.shape[1]} características",
        f"{len(np.unique(subjects))} sujetos",
        f"{np.sum(y == 0)} / {np.sum(y == 1)}",
        "",
        f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}",
        f"{cv_f1:.3f}",
        "50%",
        f"+{(cv_scores.mean() - 0.5) * 100:.1f} puntos porcentuales",
        "",
        f"{p_value:.6f} {sig_level}",
        f"{effect_size:.2f} ({effect_interpretation})",
        f"{ci_lower:.1%}",
        f"{ci_upper:.1%}",
        "Sí" if ci_lower > 0.5 else "No",
    ],
}

df_results = pd.DataFrame(results_summary)
print(df_results.to_string(index=False))

print("Bandas de frecuencia más discriminativas")
print(band_summary.round(3).to_string())


print(f"""
Resultados:
  - Precisión validada cruzadamente: {cv_scores.mean():.1%} (azar = 50%)
  - Esto es {(cv_scores.mean() - 0.5) * 100:.1f} puntos porcentuales sobre el azar
  - Valor-p = {p_value:.6f} → {"estadísticamente significativo" if p_value < 0.05 else "no estadísticamente significativo"}
  - Tamaño del efecto = {effect_size:.2f} → significancia práctica {effect_interpretation}
  - IC 95% = [{ci_lower:.1%}, {ci_upper:.1%}]
""")


results_dict = {
    "precision_cv_media": cv_scores.mean(),
    "precision_cv_std": cv_scores.std(),
    "puntaje_f1_cv": cv_f1,
    "valor_p": p_value,
    "tamaño_efecto": effect_size,
    "ic_inferior": ci_lower,
    "ic_superior": ci_upper,
    "n_muestras": X.shape[0],
    "n_caracteristicas": X.shape[1],
    "n_sujetos": len(np.unique(subjects))
}

# Guardar como JSON
with open(RESULTS_DIR / "results_summary.json", "w") as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)


