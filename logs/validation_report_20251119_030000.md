# Informe de Validación del Proyecto

**Fecha:** 2025-11-19T03:00:00.581189

## Resumen Ejecutivo

- **Estado General:** FAILED
- **Verificaciones Totales:** 7
- **Verificaciones Aprobadas:** 5
- **Verificaciones Fallidas:** 2
- **Tasa de Éxito:** 71.4%

## Detalles de Verificaciones

### Directory Structure

**Estado:** ✓ PASS

**Detalles:**
- required: 11
- found: 11
- missing: []

### Raw Data

**Estado:** ✓ PASS

**Detalles:**
- slow_files: 710
- fast_files: 706
- total_files: 1416
- unique_subjects: 45
- slow_subjects: 45
- fast_subjects: 45

### Preprocessed Data

**Estado:** ✓ PASS

**Detalles:**
- metadata_rows: 1416
- slow_dirs: 710
- fast_dirs: 706
- total_dirs: 1416
- correct_structure: True

### Graphs

**Estado:** ✓ PASS

**Detalles:**
- slow_dirs: 710
- fast_dirs: 707
- total_dirs: 1417
- correct_structure: True

### Features

**Estado:** ✗ FAIL

**Detalles:**
- n_samples: 0
- n_features: 0
- n_slow: 0
- n_fast: 0
- n_unique_subjects: 0
- feature_names_count: 0
- has_nan: False
- has_inf: False

### Images

**Estado:** ✓ PASS

**Detalles:**
- n_images: 5
- images: ['confusion_matrix.png', 'confusion matrix.png', 'feature_importance.png', 'bandsimade.png', 'output.png']

### Numerical Consistency

**Estado:** ✗ FAIL

**Detalles:**
- metadata_files: 1416
- feature_samples: 0
- slow_samples: 0
- fast_samples: 0
- all_checks_passed: False
- individual_checks: {'metadata_vs_features': False, 'balanced_classes': np.True_, 'no_nan': True, 'no_inf': True, 'valid_labels': False}

## Errores Detectados

- features
- numerical_consistency

