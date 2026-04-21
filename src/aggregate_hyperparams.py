# ===================================================================
# Código para obtener la media de los hiperparámetros para un modelo
# ===================================================================

import os
import glob
import re
from collections import defaultdict

def aggregate_hyperparameters():
    # 1. Definir rutas
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results", "optuna_hyperparams")
    
    if not os.path.exists(results_dir):
        print(f"La carpeta {results_dir} no existe. Corre Optuna primero.")
        return

    # Diccionario anidado para agrupar. Ej: model_params["PySR"]["niterations"] = [40, 60]
    model_params = defaultdict(lambda: defaultdict(list))

    # Expresión regular para capturar la línea "  - parametro: valor"
    param_pattern = re.compile(r"^\s*-\s+([a-zA-Z0-9_]+):\s+([0-9\.eE+-]+)$")

    # 2. Leer todos los TXT generados por Optuna
    archivos_txt = glob.glob(os.path.join(results_dir, "optuna_*.txt"))
    if not archivos_txt:
        print("No se encontraron archivos de resultados de Optuna.")
        return

    for filepath in archivos_txt:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        model_name = "Desconocido"
        
        # Extraer el nombre del modelo de la cabecera
        for line in lines:
            if "=== RESULTADOS OPTIMIZACIÓN:" in line:
                # Ej: === RESULTADOS OPTIMIZACIÓN: MLP_Standard en oscillator_no_noise ===
                parts = line.split("RESULTADOS OPTIMIZACIÓN:")[1].strip().split(" en ")
                model_name = parts[0].strip()
                break
                
        if model_name == "Desconocido":
            continue # Salta este archivo si no tiene el formato esperado
            
        # Extraer los parámetros
        in_params_section = False
        for line in lines:
            if "Mejores Hiperparámetros:" in line:
                in_params_section = True
                continue
                
            if in_params_section:
                match = param_pattern.match(line)
                if match:
                    param_name = match.group(1)
                    param_value_str = match.group(2)
                    
                    # Convertir a float o int
                    if "." in param_value_str or "e" in param_value_str.lower():
                        val = float(param_value_str)
                    else:
                        val = int(param_value_str)
                        
                    model_params[model_name][param_name].append(val)

    # 3. Calcular medias y guardar
    output_path = os.path.join(results_dir, "PROMEDIO_HIPERPARAMETROS.txt")
    
    with open(output_path, "w", encoding="utf-8") as out:
        header = "\n" + "="*50 + "\n PROMEDIOS DE HIPERPARÁMETROS POR MODELO \n" + "="*50
        print(header)
        out.write("=== PROMEDIOS DE HIPERPARÁMETROS POR MODELO ===\n\n")
        
        for model, params in model_params.items():
            print(f"\nModelo: {model}")
            out.write(f"\nModelo: {model}\n")
            
            for p_name, p_values in params.items():
                avg_val = sum(p_values) / len(p_values)
                
                # Si todos los valores originales eran enteros (ej: epochs = 100), redondeamos la media
                if all(isinstance(v, int) for v in p_values):
                    avg_val = int(round(avg_val))
                else:
                    # Para floats (ej: learning rate) dejamos formato científico
                    avg_val = f"{avg_val:.4e}"
                    
                print(f"  - {p_name}: {avg_val} (basado en {len(p_values)} datasets)")
                out.write(f"  - {p_name}: {avg_val} (basado en {len(p_values)} datasets)\n")
                
    print(f"\n[!] Informe generado con éxito en: {output_path}")

if __name__ == "__main__":
    aggregate_hyperparameters()