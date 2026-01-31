# Clasificación de Condiciones de Audio mediante Análisis Topológico de Conectividad EEG en Infantes

## Descripción

Este proyecto utiliza Análisis Topológico de Datos (TDA) para clasificar condiciones de audio lento y rápido basándose en patrones de conectividad cerebral en señales EEG de infantes. El análisis extrae características topológicas de grafos de conectividad funcional y utiliza Random Forest para clasificación con validación cruzada rigurosa a nivel de sujeto.

**Resultados principales:**
- Precisión de clasificación: 89.3% ± 1.7%
- Valor-p: < 0.001 (test de permutación)
- IC 95%: [92.1%, 96.4%]

## Estructura del Proyecto

```
Proyecto EEG Audio Topology/
├── tda_eeg_classification_v2.py    # Script principal de análisis
├── requirements.txt                 # Dependencias Python
├── README.md                        # Este archivo
├── informe_paper_resultados.md     # Informe académico completo
├── scripts/                         # Scripts auxiliares
│   ├── run_full_validation.py
│   └── quick_check.py
├── graphs/                          # [NO INCLUIDO] Grafos de conectividad
├── features/                        # [NO INCLUIDO] Características extraídas
├── results/                         # [NO INCLUIDO] Resultados y figuras
└── raw_data/                        # [NO INCLUIDO] Datos EEG originales
```

## Requisitos

### Software
- Python 3.10 o superior
- Sistema operativo: macOS, Linux o Windows

### Librerías
Instalar dependencias con:
```bash
pip install -r requirements.txt
```

Librerías principales:
- `numpy` - Operaciones numéricas
- `pandas` - Manipulación de datos
- `ripser` - Cálculo de homología persistente
- `scikit-learn` - Machine learning y validación
- `matplotlib` y `seaborn` - Visualización

## Datos

**IMPORTANTE:** Los datos EEG originales NO están incluidos en este repositorio por razones de privacidad. El proyecto espera la siguiente estructura de datos:

### Estructura de Directorios de Datos
```
graphs/
├── slow/                    # Grafos de audio lento
│   ├── bbXX_utYY/          # Un directorio por grabación
│   │   ├── delta_distances.npy
│   │   ├── theta_distances.npy
│   │   ├── alpha_distances.npy
│   │   ├── beta_distances.npy
│   │   └── gamma_distances.npy
│   └── ...
└── fast/                    # Grafos de audio rápido
    └── (misma estructura)
```

### Formato de Datos
Cada archivo `*_distances.npy` contiene:
- **Forma:** `(n_ventanas, n_electrodos, n_electrodos)`
- **Tipo:** Matriz de distancia (simétrica, no negativa, diagonal cero)
- **Ventanas:** 1 segundo con 75% overlap
- **Electrodos:** 47 canales

### Nombres de Archivos
Formato: `bbXX_utYY` donde:
- `bb` = identificador de sujeto
- `XX` = número de sujeto (01-45)
- `ut` = identificador de ensayo
- `YY` = número de ensayo

## Uso

### Ejecución Completa
```bash
python3 tda_eeg_classification_v2.py
```

Este script ejecuta todo el pipeline:
1. Carga matrices de distancia
2. Extrae características TDA (diagramas de persistencia)
3. Entrena clasificador Random Forest
4. Valida con GroupKFold (k=5) a nivel de sujeto
5. Realiza tests estadísticos (permutación, bootstrap)
6. Genera visualizaciones y reportes

**Tiempo estimado:** 20-30 minutos (depende de hardware)

### Verificación Rápida
```bash
python3 scripts/quick_check.py
```

### Validación Completa
```bash
python3 scripts/run_full_validation.py
```

## Metodología

### Preprocesamiento
- Filtrado en 5 bandas: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz)
- Segmentación: ventanas de 1s con 75% overlap
- Conectividad: distancia de correlación entre electrodos

### Análisis Topológico (TDA)
- **Software:** Ripser (complejos de Rips)
- **Dimensiones:** H0 (componentes conectadas), H1 (ciclos)
- **Características:** 11 por dimensión (nacimiento, muerte, persistencia, entropía)
- **Agregación:** Media y desviación estándar sobre ventanas temporales
- **Total:** 220 características (5 bandas × 22 características × 2 estadísticos)

### Clasificación
- **Algoritmo:** Random Forest (100 árboles, max_depth=10)
- **Validación:** GroupKFold (k=5) a nivel de sujeto
- **Pipeline:** StandardScaler → RandomForestClassifier
- **Métricas:** Accuracy, F1, ROC AUC, matriz de confusión

### Validación Estadística
- Test de permutación (1,000 iteraciones)
- Bootstrap confidence intervals (1,000 iteraciones)
- Tamaño del efecto (d de Cohen)

## Reproducibilidad

### Semilla Aleatoria
```python
np.random.seed(42)
RANDOM_STATE = 42
```

### Parámetros TDA
```python
MAX_DIM = 1                # Dimensión homológica máxima
MAX_EDGE_LENGTH = 2.0      # Longitud máxima de arista
```

### Parámetros de Clasificación
```python
N_SPLITS = 5               # Folds de validación cruzada
N_PERMUTATIONS = 1000      # Iteraciones test de permutación
N_BOOTSTRAP = 1000         # Iteraciones bootstrap
```

## Resultados

Los resultados se guardan automáticamente en el directorio `results/`:

- `confusion_matrix_cv.png` - Matriz de confusión validada
- `feature_importance.png` - Importancia de características
- `statistical_tests.png` - Visualización de tests estadísticos
- `subject_distribution.png` - Distribución de muestras por sujeto
- `sample_persistence_diagram.png` - Diagrama de persistencia ejemplo
- `results_summary.json` - Resumen cuantitativo completo

### Principales Hallazgos

**Rendimiento:**
- Precisión CV: 89.3% ± 1.7%
- F1 Score: 0.893
- ROC AUC: 0.951

**Significancia:**
- Valor-p (permutación): 0.000999
- Tamaño efecto (Cohen's d): 23.30
- IC 95%: [92.1%, 96.4%]

**Características Discriminativas:**
- Bandas: Beta (26.9%), Theta (23.4%), Alpha (20.6%)
- Dimensiones: H1 (54.0%), H0 (46.0%)
- Característica clave: Entropía de persistencia

## Limitaciones

- Tamaño de muestra: 45 sujetos (1,416 grabaciones)
- Sin comparación con métodos no-TDA
- Variables confusoras no controladas (edad, hora del día, etc.)
- Parámetros TDA no optimizados sistemáticamente

## Trabajo Futuro

- Comparación con análisis espectral clásico
- Exploración de dimensiones homológicas superiores (H2)
- Análisis de conectividad direccional
- Correlación con medidas conductuales
- Aplicación a detección de trastornos del desarrollo

## Citación

Si utilizas este código, por favor cita:

```
Clasificación de Condiciones de Audio mediante Análisis Topológico de Datos
de Conectividad EEG en Infantes (2025)
```

## Licencia

Este código está disponible para propósitos académicos y de investigación.

## Contacto

Para preguntas sobre el código o metodología, por favor abre un issue en este repositorio.

## Referencias

- Ripser: Ulrich Bauer (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes.
- Scikit-learn: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
- Análisis TDA en neurociencia: Giusti et al. (2015). Two's company, three (or more) is a simplex.
