# Análisis Topológico de Respuestas EEG en Infantes a Estímulos de Audio Lento y Rápido

## Resumen
Proyecto 1.b: Regresión por Grafo de Conectividad. El objetivo es evaluar si la respuesta EEG a audios
lentos presenta patrones topológicos distinguibles respecto a audios rápidos, usando conectividad funcional
entre electrodos y homología persistente.

## Datos
- 1,416 grabaciones EEG de 45 infantes (slow: 710, fast: 706).
- Archivos MATLAB `bbXX_utYY.mat` con `subeeg`, `y`, `Fs`.
- 47 electrodos válidos (ver `proyecto.pdf` para el listado completo).

## Metodología
1) Preprocesamiento por bandas de frecuencia y ventanas deslizantes.  
2) Grafos de conectividad por correlación y distancia.  
3) TDA (H0/H1) y extracción de features por banda.  
4) Clasificación con Random Forest y validación GroupKFold por sujeto.

## Estructura
- `data/`: datos crudos (`slow/`, `fast/`).  
- `preprocessed/`: ventanas por banda y archivo.  
- `graphs/`: matrices de correlación y distancia por ventana.  
- `features/`: `X.npy`, `y.npy`, `subjects.npy`, `feature_names.txt`.  
- `results/`: métricas, figuras y resúmenes.  
- `notebooks/fases/`: EDA, preprocesamiento y construcción de grafos.  
- `scripts/tda_eeg_classification_v2.py`: extracción TDA, clasificación y validación.  
- `Maria_gothe.pdf`: reporte técnico.  
- `Presentacion TDA.pdf`: presentación.  
- `proyecto.pdf`: enunciado original.

## Ejecución
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1) Preprocesamiento: `notebooks/fases/1_preprocesamiento.ipynb`  
2) Grafos: `notebooks/fases/2_graph_construction.ipynb`  
3) TDA + clasificación: `scripts/tda_eeg_classification_v2.py`
