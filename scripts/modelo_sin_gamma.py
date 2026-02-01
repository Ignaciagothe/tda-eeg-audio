"""
Modelo de clasificación TDA sin características de banda Gamma
==============================================================

Este script entrena el mismo modelo Random Forest optimizado pero
excluyendo todas las características de la banda gamma para evaluar
su impacto en la clasificación.

Autor: Generado automáticamente
Fecha: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, accuracy_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración
SEMILLA_ALEATORIA = 42
N_PARTICIONES = 5
N_BOOTSTRAP = 1000
DIRECTORIO_RESULTADOS = Path("resultados")
DIRECTORIO_RESULTADOS.mkdir(exist_ok=True)

print("=" * 70)
print("MODELO DE CLASIFICACIÓN TDA - SIN BANDA GAMMA")
print("=" * 70)

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print("\n1. Cargando datos...")

X_original = np.load('caracteristicas/X.npy')
y = np.load('caracteristicas/y.npy')
sujetos = np.load('caracteristicas/sujetos.npy')

with open('caracteristicas/nombres_caracteristicas.txt', 'r') as f:
    nombres_features = f.read().strip().split('\n')

print(f"   Datos originales: {X_original.shape[0]} muestras, {X_original.shape[1]} características")
print(f"   Sujetos únicos: {len(np.unique(sujetos))}")

# =============================================================================
# 2. FILTRAR CARACTERÍSTICAS (EXCLUIR GAMMA)
# =============================================================================
print("\n2. Filtrando características (excluyendo gamma)...")

# Identificar índices de características que NO son gamma
indices_sin_gamma = [i for i, nombre in enumerate(nombres_features) if 'gamma' not in nombre.lower()]
indices_gamma = [i for i, nombre in enumerate(nombres_features) if 'gamma' in nombre.lower()]

# Filtrar X
X = X_original[:, indices_sin_gamma]
nombres_features_filtrados = [nombres_features[i] for i in indices_sin_gamma]

print(f"   Características gamma excluidas: {len(indices_gamma)}")
print(f"   Características restantes: {len(indices_sin_gamma)}")
print(f"   X filtrado: {X.shape}")

# Mostrar bandas restantes
bandas_restantes = set([nombre.split('_')[0] for nombre in nombres_features_filtrados])
print(f"   Bandas incluidas: {sorted(bandas_restantes)}")

# =============================================================================
# 3. CONFIGURAR CROSS-VALIDATION
# =============================================================================
print("\n3. Configurando GroupKFold...")

gkf = GroupKFold(n_splits=N_PARTICIONES)

# Verificar separación de sujetos
print(f"   Verificando separación de sujetos:")
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=sujetos), 1):
    train_sujetos = set(sujetos[train_idx])
    test_sujetos = set(sujetos[test_idx])
    overlap = len(train_sujetos.intersection(test_sujetos))
    print(f"   Fold {fold_idx}: Train={len(train_sujetos)} sujetos, Test={len(test_sujetos)} sujetos, Overlap={overlap} {'✓' if overlap == 0 else '❌'}")

# =============================================================================
# 4. GRID SEARCH PARA OPTIMIZACIÓN
# =============================================================================
print("\n4. Ejecutando Grid Search...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

modelo_base = RandomForestClassifier(random_state=SEMILLA_ALEATORIA, n_jobs=-1)

print(f"   Espacio de búsqueda: 216 combinaciones")
print(f"   Ejecutando GridSearchCV...")

grid_search = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X, y, groups=sujetos)

print(f"\n   Mejores hiperparámetros:")
for param, value in grid_search.best_params_.items():
    print(f"     {param}: {value}")
print(f"   Mejor precisión CV: {grid_search.best_score_:.4f} ({grid_search.best_score_:.1%})")

# =============================================================================
# 5. EVALUACIÓN DEL MODELO OPTIMIZADO
# =============================================================================
print("\n5. Evaluando modelo optimizado...")

modelo_rf = grid_search.best_estimator_

# Cross-validation scores
puntajes_cv = cross_val_score(modelo_rf, X, y, groups=sujetos, cv=gkf, scoring="accuracy")
print(f"\n   Resultados de Validación Cruzada:")
print(f"     Precisión por fold: {puntajes_cv}")
print(f"     Precisión media: {puntajes_cv.mean():.4f} ± {puntajes_cv.std():.4f}")
print(f"     Mín/Máx: {puntajes_cv.min():.4f} / {puntajes_cv.max():.4f}")

# Predicciones CV
y_pred_cv = cross_val_predict(modelo_rf, X, y, groups=sujetos, cv=gkf)
y_proba_cv = cross_val_predict(modelo_rf, X, y, groups=sujetos, cv=gkf, method="predict_proba")

# Métricas
f1_cv = f1_score(y, y_pred_cv, average="weighted")
auc_cv = roc_auc_score(y, y_proba_cv[:, 1])

print(f"\n   Métricas adicionales:")
print(f"     F1 Score (weighted): {f1_cv:.4f}")
print(f"     ROC AUC: {auc_cv:.4f}")

print("\n   Reporte de Clasificación:")
print(classification_report(y, y_pred_cv, target_names=["Lento", "Rápido"]))

# =============================================================================
# 6. BOOTSTRAP PARA INTERVALO DE CONFIANZA
# =============================================================================
print("\n6. Ejecutando Bootstrap para IC 95%...")

def bootstrap_puntaje_cv(X, y, grupos, modelo, cv, n_bootstrap=1000, semilla=42):
    """Bootstrap con IDs de grupo originales (corregido)."""
    np.random.seed(semilla)
    puntajes_bootstrap = []

    sujetos_unicos = np.unique(grupos)
    n_sujetos = len(sujetos_unicos)

    for i in tqdm(range(n_bootstrap), desc="   Bootstrap"):
        sujetos_boot = np.random.choice(sujetos_unicos, size=n_sujetos, replace=True)

        indices_boot = []
        nuevos_grupos = []
        for j, sujeto in enumerate(sujetos_boot):
            indices_sujeto = np.where(grupos == sujeto)[0]
            indices_boot.extend(indices_sujeto)
            # Mantener ID de grupo original (CORREGIDO)
            nuevos_grupos.extend([sujeto] * len(indices_sujeto))

        indices_boot = np.array(indices_boot)
        nuevos_grupos = np.array(nuevos_grupos)

        X_boot = X[indices_boot]
        y_boot = y[indices_boot]

        if len(np.unique(y_boot)) < 2:
            continue
        if len(np.unique(nuevos_grupos)) < cv.n_splits:
            continue

        try:
            puntajes_boot = cross_val_score(
                modelo, X_boot, y_boot, groups=nuevos_grupos, cv=cv, scoring="accuracy"
            )
            puntajes_bootstrap.append(puntajes_boot.mean())
        except Exception:
            continue

    return np.array(puntajes_bootstrap)

precisiones_bootstrap = bootstrap_puntaje_cv(
    X, y, sujetos, modelo_rf, gkf,
    n_bootstrap=N_BOOTSTRAP,
    semilla=SEMILLA_ALEATORIA
)

ic_inferior = np.percentile(precisiones_bootstrap, 2.5)
ic_superior = np.percentile(precisiones_bootstrap, 97.5)

print(f"\n   Resultados Bootstrap ({len(precisiones_bootstrap)} iteraciones):")
print(f"     Media bootstrap: {precisiones_bootstrap.mean():.4f}")
print(f"     IC 95%: [{ic_inferior:.4f}, {ic_superior:.4f}]")
print(f"     IC 95%: [{ic_inferior:.1%}, {ic_superior:.1%}]")

media_obs = puntajes_cv.mean()
if ic_inferior <= media_obs <= ic_superior:
    print(f"     ✓ Valor observado ({media_obs:.1%}) está dentro del IC")
else:
    print(f"     ⚠ Valor observado ({media_obs:.1%}) fuera del IC")

# =============================================================================
# 7. COMPARACIÓN CON MODELO COMPLETO
# =============================================================================
print("\n7. Comparación con modelo completo (con gamma)...")

# Cargar resultados del modelo completo si existen
try:
    bootstrap_completo = np.load('caracteristicas/bootstrap_scores.npy')
    media_completo = bootstrap_completo.mean()
    ic_inf_completo = np.percentile(bootstrap_completo, 2.5)
    ic_sup_completo = np.percentile(bootstrap_completo, 97.5)

    print(f"\n   Modelo COMPLETO (con gamma):")
    print(f"     Precisión media: ~81.6%")
    print(f"     IC 95%: [{ic_inf_completo:.1%}, {ic_sup_completo:.1%}]")

    print(f"\n   Modelo SIN GAMMA:")
    print(f"     Precisión media: {puntajes_cv.mean():.1%}")
    print(f"     IC 95%: [{ic_inferior:.1%}, {ic_superior:.1%}]")

    diferencia = (puntajes_cv.mean() - 0.816) * 100
    print(f"\n   Diferencia: {diferencia:+.2f} puntos porcentuales")

    if diferencia < -2:
        print(f"   → La banda gamma CONTRIBUYE significativamente al modelo")
    elif diferencia > 2:
        print(f"   → El modelo MEJORA sin gamma (posible ruido)")
    else:
        print(f"   → La banda gamma tiene POCO IMPACTO en el rendimiento")

except FileNotFoundError:
    print("   (No se encontraron resultados del modelo completo para comparar)")

# =============================================================================
# 8. VISUALIZACIÓN
# =============================================================================
print("\n8. Generando visualizaciones...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión
mc = confusion_matrix(y, y_pred_cv)
sns.heatmap(
    mc,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Lento", "Rápido"],
    yticklabels=["Lento", "Rápido"],
    ax=axes[0],
    annot_kws={"size": 16}
)
axes[0].set_xlabel("Predicción", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Real", fontsize=12, fontweight="bold")
axes[0].set_title("Matriz de Confusión (Sin Gamma)", fontsize=14, fontweight="bold")

# Distribución bootstrap
axes[1].hist(precisiones_bootstrap, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[1].axvline(media_obs, color='red', linestyle='--', linewidth=2, label=f'Observada ({media_obs:.1%})')
axes[1].axvline(ic_inferior, color='orange', linestyle=':', linewidth=2, label=f'IC 95%')
axes[1].axvline(ic_superior, color='orange', linestyle=':', linewidth=2)
axes[1].axvspan(ic_inferior, ic_superior, alpha=0.2, color='orange')
axes[1].axvline(0.5, color='green', linestyle='-', linewidth=1, label='Azar (50%)')
axes[1].set_xlabel("Precisión CV", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Densidad", fontsize=12, fontweight="bold")
axes[1].set_title("Bootstrap IC 95% (Sin Gamma)", fontsize=14, fontweight="bold")
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig(DIRECTORIO_RESULTADOS / "modelo_sin_gamma.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"   Figura guardada: {DIRECTORIO_RESULTADOS / 'modelo_sin_gamma.png'}")

# =============================================================================
# 9. IMPORTANCIA DE CARACTERÍSTICAS
# =============================================================================
print("\n9. Analizando importancia de características...")

# Entrenar modelo final
modelo_rf.fit(X, y)
importancias = modelo_rf.feature_importances_

# Top 15 características más importantes
indices_top = np.argsort(importancias)[::-1][:15]

print(f"\n   Top 15 características más importantes (sin gamma):")
for i, idx in enumerate(indices_top, 1):
    print(f"     {i:2d}. {nombres_features_filtrados[idx]}: {importancias[idx]:.4f}")

# Visualización de importancia
fig, ax = plt.subplots(figsize=(10, 8))
top_nombres = [nombres_features_filtrados[i] for i in indices_top]
top_importancias = importancias[indices_top]

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_nombres)))
bars = ax.barh(range(len(top_nombres)), top_importancias, color=colors)
ax.set_yticks(range(len(top_nombres)))
ax.set_yticklabels(top_nombres)
ax.invert_yaxis()
ax.set_xlabel("Importancia", fontsize=12, fontweight="bold")
ax.set_title("Top 15 Características Más Importantes (Sin Gamma)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(DIRECTORIO_RESULTADOS / "importancia_sin_gamma.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"   Figura guardada: {DIRECTORIO_RESULTADOS / 'importancia_sin_gamma.png'}")

# =============================================================================
# 10. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL - MODELO SIN GAMMA")
print("=" * 70)

print(f"""
Configuración:
  - Características: {X.shape[1]} (excluidas {len(indices_gamma)} de gamma)
  - Bandas incluidas: {', '.join(sorted(bandas_restantes))}
  - Muestras: {X.shape[0]}
  - Sujetos: {len(np.unique(sujetos))}

Resultados del Modelo Optimizado:
  - Precisión CV: {puntajes_cv.mean():.4f} ± {puntajes_cv.std():.4f} ({puntajes_cv.mean():.1%})
  - F1 Score: {f1_cv:.4f}
  - ROC AUC: {auc_cv:.4f}
  - IC 95% Bootstrap: [{ic_inferior:.1%}, {ic_superior:.1%}]

Mejores Hiperparámetros:
""")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print("\n" + "=" * 70)
print("Ejecución completada exitosamente")
print("=" * 70)

# Guardar resultados
np.save('caracteristicas/bootstrap_sin_gamma.npy', precisiones_bootstrap)
print(f"\nResultados guardados en: caracteristicas/bootstrap_sin_gamma.npy")
