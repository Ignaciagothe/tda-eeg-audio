# TDA-EEG-Audio: Clasificación Topológica de Respuestas EEG en Bebés

## Resumen

Análisis de respuestas EEG de bebés (3-5 meses) a estímulos auditivos usando **Topological Data Analysis (TDA)**.
El objetivo es clasificar si el bebé escucha audio en velocidad normal (slow) o acelerada (fast) basándose
en las características topológicas de la conectividad funcional cerebral.

### Resultados Principales

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 80.8% ± 2.0% |
| **Balanced Accuracy** | 80.8% |
| **ROC-AUC** | 0.887 |
| **p-value** | 0.001 (***) |
| **95% CI** | [79.1%, 82.4%] |
| **Mejor banda** | Gamma (60% importancia) |

---

> Nota: estos valores corresponden a resultados históricos. Tras las correcciones
> metodológicas recientes (nested CV, ventanas igualadas, bandas consistentes),
> se deben **re‑ejecutar** los scripts para obtener métricas actualizadas.

## Datos

- **45 bebés** (bb01-bb84, con gaps)
- **1,416 grabaciones** EEG (710 slow + 706 fast)
- **47 electrodos** válidos (de 65 totales)
- **EEG**: 250 Hz | **Audio**: 44,100 Hz

Archivos MATLAB: `bbXX_utYY.mat` con:

- `subeeg`: señal EEG (electrodos × muestras)
- `y`: señal de audio
- `Fs`: frecuencia de muestreo del audio

---

## Estructura del Proyecto

```
tda-eeg-audio/
│
├── data/                          # Datos crudos (.mat)
│   ├── slow/                      # 710 grabaciones velocidad normal
│   └── fast/                      # 706 grabaciones aceleradas
│
├── preprocessed/                  # Datos preprocesados
│   ├── slow/                      # Ventanas filtradas por banda
│   ├── fast/
│   └── preprocessing_metadata.csv
│
├── graphs/                        # Grafos de conectividad
│   ├── slow/                      # Matrices correlación/distancia
│   └── fast/
│
├── features/                      # Features TDA extraídas
│   ├── X.npy                      # (1416, 220) features
│   ├── y.npy                      # etiquetas (0=slow, 1=fast)
│   ├── subjects.npy               # IDs de sujetos
│   ├── feature_names.txt          # nombres de features
│   └── filenames.txt              # nombres de archivos
│
├── results/                       # Resultados y visualizaciones
│   ├── results_publication.json   #  Resultados finales
│   ├── analysis_publication.png   #  Figura principal
│   ├── analysis_publication.pdf   #  Figura para paper
│   └── old/                       # Resultados anteriores
│
├── scripts/                       # Scripts de Python
│   ├── tda_analysis_publication.py    # SCRIPT PRINCIPAL (nested CV)
│   ├── tda_eeg_classification_v2.py   # Extracción de features TDA (ventanas igualadas)
│   ├── tda_synchronization_analysis.py # Sincronización EEG-Audio
│   ├── tda_analysis_corrected.py      # (legacy)
│   └── tda_analysis_final.py          # (legacy)
│
├── notebooks/                     # Jupyter notebooks
│    ├── 0_eda.ipynb                # Análisis exploratorio
│    ├── 1_preprocesamiento.ipynb   # Filtrado y ventanas
│    ├── 2_graph_construction.ipynb # Construcción de grafos
│
├── docs/                          # Documentación y paper
│   ├── paper_final.tex            # Paper en LaTeX
│   ├── paper_final.pdf            # Paper compilado
│   ├── proyecto.pdf            
│   └── *.md                 
│          
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo

```

---

## Archivos Vigentes (usar estos)

### Scripts Principales

| Archivo | Descripción |
|---------|-------------|
| `scripts/tda_analysis_publication.py` | **Script principal** - Análisis completo (nested CV estratificado por sujeto) |
| `scripts/tda_eeg_classification_v2.py` | Extracción de features TDA desde grafos (ventanas igualadas) |
| `scripts/tda_synchronization_analysis.py` | Análisis de sincronización EEG‑Audio (envelopes) |

### Notebooks del Pipeline

| Archivo | Descripción |
|---------|-------------|
| `notebooks/fases/0_eda.ipynb` | Análisis exploratorio de datos |
| `notebooks/fases/1_preprocesamiento.ipynb` | Filtrado por bandas y ventanas |
| `notebooks/fases/2_graph_construction.ipynb` | Construcción de grafos de conectividad |

### Resultados Finales

| Archivo | Descripción |
|---------|-------------|
| `results/results_publication.json` | Métricas y resultados completos |
| `results/analysis_publication.png` | Figura principal (300 DPI) |
| `results/analysis_publication.pdf` | Figura vectorial para paper |

---

## Metodología

### 1. Preprocesamiento

- Filtrado Butterworth (orden 4) en 5 bandas: Delta, Theta, Alpha, Beta, Gamma
- Ventanas deslizantes: 1s con 75% overlap
- Solo 47 electrodos válidos (parte superior de la cabeza)

### 2. Grafos de Conectividad

- Correlación de Pearson entre pares de electrodos
- Matriz de distancia (métrica): `d = sqrt(2*(1 - r))`  
  (opción alternativa: `1 - |r|`, **no métrica**)
- Matrices 47×47 por ventana

### 3. TDA (Topological Data Analysis)

- Homología persistente H0 (componentes conectadas) y H1 (ciclos)
- Features: birth, death, persistence, entropy, etc.
- Agregación temporal: mean y std por grabación

### 4. Clasificación

- Random Forest con `class_weight='balanced'`
- Validación: 5-fold **StratifiedGroupKFold** por sujeto (balance por clase)
- Selección de `k` e hiperparámetros con **nested CV**
- Tests estadísticos: Permutation test (within-subject) + Bootstrap BCa CI

---

## Ejecución

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar análisis principal
python scripts/tda_analysis_publication.py
```

### Pipeline completo (si se tienen datos crudos)

1. `notebooks/fases/1_preprocesamiento.ipynb` → genera `preprocessed/`
2. `notebooks/fases/2_graph_construction.ipynb` → genera `graphs/`
3. `scripts/tda_eeg_classification_v2.py` → genera `features/` + `features/metadata.csv`
4. `scripts/tda_analysis_publication.py` → genera `results/`
5. (Opcional) `scripts/tda_synchronization_analysis.py` → genera `results/eeg_audio_*`

---

## Dependencias Principales

- numpy, pandas, scipy
- scikit-learn
- ripser (TDA)
- matplotlib, seaborn
- tqdm

---

## Hallazgos Clave

1. **Banda Gamma domina**: 60% de la importancia predictiva
2. **H1 (ciclos) > H0**: La estructura de ciclos en conectividad es más informativa
3. **Feature más importante**: `gamma_h1_persistence_entropy_std` (20.9%)
4. **Generalización robusta**: p < 0.001 con GroupKFold por sujeto

---

## Archivos Legacy (no usar para resultados finales)

- `scripts/tda_analysis_final.py` - Script legacy (sin nested CV)
- `scripts/tda_analysis_corrected.py` - Script legacy (sin nested CV)
- `notebooks/legacy/*` - Notebooks de desarrollo
- `results/old/*` - Imágenes anteriores

---

## Autores

- Ignacia Gothe

## Referencias

- Edelsbrunner, H., & Harer, J. (2010). Computational Topology.
- Carlsson, G. (2009). Topology and data. Bulletin of the AMS.
