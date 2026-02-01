# Análisis Topológico de Respuestas EEG en Infantes a Estímulos de Audio Lento y Rápido

## Proyecto 1.b (regresión por grafo de conectividad)
- Datos: series de tiempo = grabaciones audios + EEG de respuesta de varios niños al escuchar
- Pregunta Cientifica: verificar si audios lentos inducen mejor semejanza en el EEG que audios rápidos
- Metodos baseline para comparar: procesamiento por TRF (no parte del proyecto)

#### Propuesta base de pipeline:
1. Preprocesamiento:
    a. Subdividir canales EEG por bandas de frecuencia
    b. Calcular correlacion por sliding window etre electrodos: se crea serie de tiempo de grafos pesados
2. Por cada banda de frecuencia: aplicar filtración basado en los pesos del grafo por sliding window en el tiempo para crear serie
de tiempo de marcadores topológicos del EEG
3. Entrenar clasificador para asignar un matching [patron EEG]<->[patron audio] (a) exitoso en un subconjunto de datos de entrenamiento (b) y que "generalice" bien a (osea tenga poco error al aplicar a los) datos dejados de lado, de "testeo"
4. Incluir estadisticas de exito del clasificador y  estadisticas de validez

## Estructura del Proyecto
```
tda-eeg-audio % tree -L 2
.
├── README.md
├── data
│   ├── fast.zip
│   └── slow.zip
├── features
│   ├── X.npy
│   ├── feature_names.txt
│   ├── filenames.txt
│   ├── subjects.npy
│   └── y.npy
├── logs
│   ├── validation_report_20251119_030000.json
│   └── validation_report_20251119_030000.md
├── notebooks
│   ├── fases
│   ├── main.ipynb
│   └── topological_features_classification.ipynb
├── preprocessed
│   ├── fast
│   └── slow
├── requirements.txt
├── results
│   ├── confusion_matrix_cv.png
│   ├── feature_importance.png
│   ├── imagenes
│   ├── results_summary.json
│   ├── sample_persistence_diagram.png
│   ├── statistical_tests.png
│   └── subject_distribution.png
└── scripts
    ├── run_full_validation.py
    ├── script_dia_final.py
    └── tda_eeg_classification_v2.py
```


Instalar dependencias con:
```bash
pip install -r requirements.txt
```



### Nombres de Archivos
Formato: `bbXX_utYY` donde:
- `bb` = identificador de sujeto
- `XX` = número de sujeto (01-45)
- `ut` = identificador de ensayo
- `YY` = número de ensayo
