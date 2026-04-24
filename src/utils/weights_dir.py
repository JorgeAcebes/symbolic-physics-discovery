# Script para copiar las weights a una misma carpeta

import os
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
SOURCE_RESULTS = os.path.join(PROJECT_ROOT, "results")
DESTINATION_DIR = os.path.join(SOURCE_RESULTS, "weights")
os.makedirs(DESTINATION_DIR, exist_ok=True)

def collect_weights():
    # 1. Asegurar la existencia de la carpeta de destino
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    print(f"Carpeta de destino: {DESTINATION_DIR}")

    # 2. Buscar todos los archivos que terminen en _weights.json de forma recursiva
    # Esto incluye subcarpetas como PySR, GPLearn, MLP_*, etc.
    search_pattern = os.path.join(SOURCE_RESULTS, "**", "*_weights.json")
    weight_files = glob.glob(search_pattern, recursive=True)

    if not weight_files:
        print("No se encontraron archivos de pesos.")
        return

    print(f"Encontrados {len(weight_files)} archivos. Iniciando copia...")

    # 3. Copiar archivos evitando la autorreferencia si ya están en la carpeta destino
    count = 0
    for file_path in weight_files:
        if DESTINATION_DIR in os.path.abspath(file_path):
            continue
            
        file_name = os.path.basename(file_path)
        shutil.copy2(file_path, os.path.join(DESTINATION_DIR, file_name))
        count += 1

    print(f"Éxito. Se han copiado {count} archivos a {DESTINATION_DIR}")

if __name__ == "__main__":
    collect_weights()