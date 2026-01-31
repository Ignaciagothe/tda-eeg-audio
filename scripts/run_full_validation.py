#!/usr/bin/env python3
"""
Script de Validación Completa del Proyecto EEG-Audio con Análisis Topológico
Ejecuta todas las verificaciones y genera un informe detallado

Uso:
    python3 run_full_validation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import json

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
FEATURES_DIR = PROJECT_ROOT / "features"
IMAGENES_DIR = PROJECT_ROOT / "imagenes"
OUTPUT_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR.mkdir(exist_ok=True)


class ProjectValidator:
    """Validador completo del proyecto"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
    def log(self, message, level="INFO"):
        """Log con timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def add_check(self, name, passed, details=None):
        """Registrar resultado de verificación"""
        self.results["checks"][name] = {
            "passed": passed,
            "details": details
        }
        if not passed:
            self.results["errors"].append(name)
            
    def check_directory_structure(self):
        """Paso 1: Verificar estructura de directorios"""
        self.log("Verificando estructura de directorios...")
        
        required_dirs = {
            "data": DATA_DIR,
            "data/slow": DATA_DIR / "slow",
            "data/fast": DATA_DIR / "fast",
            "preprocessed": PREPROCESSED_DIR,
            "preprocessed/slow": PREPROCESSED_DIR / "slow",
            "preprocessed/fast": PREPROCESSED_DIR / "fast",
            "graphs": GRAPHS_DIR,
            "graphs/slow": GRAPHS_DIR / "slow",
            "graphs/fast": GRAPHS_DIR / "fast",
            "features": FEATURES_DIR,
            "imagenes": IMAGENES_DIR
        }
        
        missing_dirs = []
        for name, path in required_dirs.items():
            if not path.exists():
                missing_dirs.append(name)
                
        passed = len(missing_dirs) == 0
        self.add_check("directory_structure", passed, {
            "required": len(required_dirs),
            "found": len(required_dirs) - len(missing_dirs),
            "missing": missing_dirs
        })
        
        if passed:
            self.log("✓ Estructura de directorios completa")
        else:
            self.log(f"✗ Directorios faltantes: {missing_dirs}", "ERROR")
            
        return passed
        
    def check_raw_data(self):
        """Paso 2: Verificar datos crudos"""
        self.log("Verificando datos crudos...")
        
        slow_files = list((DATA_DIR / "slow").glob("*.mat"))
        fast_files = list((DATA_DIR / "fast").glob("*.mat"))
        
        # Extraer sujetos
        slow_subjects = set([f.stem.split("_")[0] for f in slow_files])
        fast_subjects = set([f.stem.split("_")[0] for f in fast_files])
        all_subjects = slow_subjects | fast_subjects
        
        details = {
            "slow_files": len(slow_files),
            "fast_files": len(fast_files),
            "total_files": len(slow_files) + len(fast_files),
            "unique_subjects": len(all_subjects),
            "slow_subjects": len(slow_subjects),
            "fast_subjects": len(fast_subjects)
        }
        
        # Verificar que hay archivos
        passed = len(slow_files) > 0 and len(fast_files) > 0
        self.add_check("raw_data", passed, details)
        
        if passed:
            self.log(f"✓ Datos crudos: {len(slow_files)} slow, {len(fast_files)} fast, {len(all_subjects)} sujetos")
        else:
            self.log("✗ Datos crudos incompletos", "ERROR")
            
        return passed
        
    def check_preprocessed_data(self):
        """Paso 4: Verificar datos preprocesados"""
        self.log("Verificando datos preprocesados...")
        
        # Verificar metadata
        metadata_file = PREPROCESSED_DIR / "preprocessing_metadata.csv"
        if not metadata_file.exists():
            self.add_check("preprocessed_data", False, {"error": "Metadata file missing"})
            self.log("✗ Archivo de metadata faltante", "ERROR")
            return False
            
        df_meta = pd.read_csv(metadata_file)
        
        # Verificar directorios
        slow_dirs = list((PREPROCESSED_DIR / "slow").iterdir())
        fast_dirs = list((PREPROCESSED_DIR / "fast").iterdir())
        
        # Verificar estructura de archivos (sample)
        sample_dir = slow_dirs[0] if slow_dirs else None
        has_correct_structure = False
        if sample_dir and sample_dir.is_dir():
            expected_files = [
                "delta.npy", "theta.npy", "alpha.npy", "beta.npy", "gamma.npy",
                "window_times.npy", "audio.npy"
            ]
            has_correct_structure = all((sample_dir / f).exists() for f in expected_files)
            
        details = {
            "metadata_rows": len(df_meta),
            "slow_dirs": len(slow_dirs),
            "fast_dirs": len(fast_dirs),
            "total_dirs": len(slow_dirs) + len(fast_dirs),
            "correct_structure": has_correct_structure
        }
        
        passed = (len(df_meta) > 0 and 
                  len(slow_dirs) > 0 and 
                  len(fast_dirs) > 0 and 
                  has_correct_structure)
        
        self.add_check("preprocessed_data", passed, details)
        
        if passed:
            self.log(f"✓ Datos preprocesados: {len(slow_dirs)} slow, {len(fast_dirs)} fast")
        else:
            self.log("✗ Datos preprocesados incompletos", "ERROR")
            
        return passed
        
    def check_graphs(self):
        """Paso 6: Verificar grafos de conectividad"""
        self.log("Verificando grafos de conectividad...")
        
        slow_dirs = list((GRAPHS_DIR / "slow").iterdir())
        fast_dirs = list((GRAPHS_DIR / "fast").iterdir())
        
        # Verificar estructura de archivos (sample)
        sample_dir = slow_dirs[0] if slow_dirs else None
        has_correct_files = False
        if sample_dir and sample_dir.is_dir():
            expected_patterns = ["_correlations.npy", "_distances.npy"]
            bands = ["delta", "theta", "alpha", "beta", "gamma"]
            expected_files = [f"{band}{pattern}" for band in bands for pattern in expected_patterns]
            has_correct_files = all((sample_dir / f).exists() for f in expected_files)
            
        details = {
            "slow_dirs": len(slow_dirs),
            "fast_dirs": len(fast_dirs),
            "total_dirs": len(slow_dirs) + len(fast_dirs),
            "correct_structure": has_correct_files
        }
        
        passed = (len(slow_dirs) > 0 and 
                  len(fast_dirs) > 0 and 
                  has_correct_files)
        
        self.add_check("graphs", passed, details)
        
        if passed:
            self.log(f"✓ Grafos: {len(slow_dirs)} slow, {len(fast_dirs)} fast")
        else:
            self.log("✗ Grafos incompletos", "ERROR")
            
        return passed
        
    def check_features(self):
        """Paso 8: Verificar características extraídas"""
        self.log("Verificando características extraídas...")
        
        required_files = {
            "X.npy": FEATURES_DIR / "X.npy",
            "y.npy": FEATURES_DIR / "y.npy",
            "subjects.npy": FEATURES_DIR / "subjects.npy",
            "feature_names.txt": FEATURES_DIR / "feature_names.txt",
            "filenames.txt": FEATURES_DIR / "filenames.txt"
        }
        
        missing_files = [name for name, path in required_files.items() if not path.exists()]
        
        if missing_files:
            self.add_check("features", False, {"missing_files": missing_files})
            self.log(f"✗ Archivos de características faltantes: {missing_files}", "ERROR")
            return False
            
        # Cargar y verificar dimensiones
        X = np.load(FEATURES_DIR / "X.npy")
        y = np.load(FEATURES_DIR / "y.npy")
        subjects = np.load(FEATURES_DIR / "subjects.npy")
        
        with open(FEATURES_DIR / "feature_names.txt") as f:
            feature_names = [line.strip() for line in f]
            
        details = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_slow": int((y == 0).sum()),
            "n_fast": int((y == 1).sum()),
            "n_unique_subjects": len(np.unique(subjects)),
            "feature_names_count": len(feature_names),
            "has_nan": bool(np.isnan(X).any()),
            "has_inf": bool(np.isinf(X).any())
        }
        
        passed = (X.shape[0] > 0 and 
                  X.shape[1] == 180 and 
                  X.shape[0] == y.shape[0] and
                  not details["has_nan"] and
                  not details["has_inf"])
        
        self.add_check("features", passed, details)
        
        if passed:
            self.log(f"✓ Características: {X.shape[0]} muestras, {X.shape[1]} features, {len(np.unique(subjects))} sujetos")
        else:
            self.log("✗ Características con problemas", "ERROR")
            
        return passed
        
    def check_images(self):
        """Paso 10: Verificar imágenes generadas"""
        self.log("Verificando visualizaciones...")
        
        image_files = list(IMAGENES_DIR.glob("*.png"))
        
        details = {
            "n_images": len(image_files),
            "images": [f.name for f in image_files]
        }
        
        passed = len(image_files) > 0
        self.add_check("images", passed, details)
        
        if passed:
            self.log(f"✓ Visualizaciones: {len(image_files)} imágenes")
        else:
            self.log("⚠ Sin visualizaciones generadas", "WARNING")
            
        return passed
        
    def check_numerical_consistency(self):
        """Paso 12: Verificar consistencia numérica"""
        self.log("Verificando consistencia numérica...")
        
        try:
            # Cargar metadata de preprocesamiento
            df_meta = pd.read_csv(PREPROCESSED_DIR / "preprocessing_metadata.csv")
            
            # Cargar características
            X = np.load(FEATURES_DIR / "X.npy")
            y = np.load(FEATURES_DIR / "y.npy")
            
            # Verificar consistencias
            checks = {
                "metadata_vs_features": abs(len(df_meta) - len(X)) <= 20,  # Tolerancia por archivos problemáticos
                "balanced_classes": abs((y == 0).sum() - (y == 1).sum()) < 50,
                "no_nan": not np.isnan(X).any(),
                "no_inf": not np.isinf(X).any(),
                "valid_labels": set(np.unique(y)) == {0, 1}
            }
            
            details = {
                "metadata_files": len(df_meta),
                "feature_samples": len(X),
                "slow_samples": int((y == 0).sum()),
                "fast_samples": int((y == 1).sum()),
                "all_checks_passed": all(checks.values()),
                "individual_checks": checks
            }
            
            passed = all(checks.values())
            self.add_check("numerical_consistency", passed, details)
            
            if passed:
                self.log("✓ Consistencia numérica verificada")
            else:
                failed_checks = [k for k, v in checks.items() if not v]
                self.log(f"✗ Inconsistencias: {failed_checks}", "ERROR")
                
            return passed
            
        except Exception as e:
            self.add_check("numerical_consistency", False, {"error": str(e)})
            self.log(f"✗ Error en verificación numérica: {e}", "ERROR")
            return False
            
    def generate_summary(self):
        """Generar resumen de validación"""
        total_checks = len(self.results["checks"])
        passed_checks = sum(1 for c in self.results["checks"].values() if c["passed"])
        
        self.results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "overall_status": "PASSED" if passed_checks == total_checks else "FAILED"
        }
        
    def save_report(self):
        """Guardar informe de validación"""
        # JSON report
        json_file = OUTPUT_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"✓ Informe JSON guardado: {json_file}")
        
        # Markdown report
        md_file = OUTPUT_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self.generate_markdown_report(md_file)
        self.log(f"✓ Informe Markdown guardado: {md_file}")
        
    def generate_markdown_report(self, filepath):
        """Generar informe en formato Markdown"""
        with open(filepath, 'w') as f:
            f.write("# Informe de Validación del Proyecto\n\n")
            f.write(f"**Fecha:** {self.results['timestamp']}\n\n")
            
            # Resumen
            summary = self.results['summary']
            f.write("## Resumen Ejecutivo\n\n")
            f.write(f"- **Estado General:** {summary['overall_status']}\n")
            f.write(f"- **Verificaciones Totales:** {summary['total_checks']}\n")
            f.write(f"- **Verificaciones Aprobadas:** {summary['passed_checks']}\n")
            f.write(f"- **Verificaciones Fallidas:** {summary['failed_checks']}\n")
            f.write(f"- **Tasa de Éxito:** {summary['success_rate']:.1%}\n\n")
            
            # Detalles de verificaciones
            f.write("## Detalles de Verificaciones\n\n")
            for name, check in self.results['checks'].items():
                status = "✓ PASS" if check['passed'] else "✗ FAIL"
                f.write(f"### {name.replace('_', ' ').title()}\n\n")
                f.write(f"**Estado:** {status}\n\n")
                if check['details']:
                    f.write("**Detalles:**\n")
                    for key, value in check['details'].items():
                        f.write(f"- {key}: {value}\n")
                f.write("\n")
                
            # Errores
            if self.results['errors']:
                f.write("## Errores Detectados\n\n")
                for error in self.results['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")
                
    def run_all_checks(self):
        """Ejecutar todas las verificaciones"""
        self.log("=" * 60)
        self.log("INICIANDO VALIDACIÓN COMPLETA DEL PROYECTO")
        self.log("=" * 60)
        
        checks = [
            ("Estructura de Directorios", self.check_directory_structure),
            ("Datos Crudos", self.check_raw_data),
            ("Datos Preprocesados", self.check_preprocessed_data),
            ("Grafos de Conectividad", self.check_graphs),
            ("Características Extraídas", self.check_features),
            ("Visualizaciones", self.check_images),
            ("Consistencia Numérica", self.check_numerical_consistency)
        ]
        
        for name, check_func in checks:
            self.log(f"\n{name}...")
            try:
                check_func()
            except Exception as e:
                self.log(f"✗ Error inesperado en {name}: {e}", "ERROR")
                self.add_check(name.lower().replace(" ", "_"), False, {"error": str(e)})
                
        self.log("\n" + "=" * 60)
        self.generate_summary()
        
        # Mostrar resumen
        summary = self.results['summary']
        self.log(f"RESUMEN: {summary['passed_checks']}/{summary['total_checks']} verificaciones aprobadas")
        self.log(f"ESTADO: {summary['overall_status']}")
        self.log("=" * 60)
        
        # Guardar informe
        self.save_report()
        
        return summary['overall_status'] == "PASSED"


def main():
    """Función principal"""
    validator = ProjectValidator()
    success = validator.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
