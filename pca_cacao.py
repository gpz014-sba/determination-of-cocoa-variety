import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = r"C:\Users\USER\Desktop\Giancarlo\Muestras de cacao"

VARIEDADES = [
    "Cacao - Mazamari",
    "Cacao Amazonas",
    "CACAO CHUNCHO",
    "Cacao Cusco 1",
    "Cacao Mazamari 2",
    "Cacao Piura",
    "Cacao San Martín",
    "CACAO SATIPO"
]

EXT = "*.xlsx"

OUTPUT_ORIG = r"C:\Users\USER\Desktop\Giancarlo\DATASETS_ORIGINALES"
OUTPUT_SNV  = r"C:\Users\USER\Desktop\Giancarlo\DATASETS_SNV"

os.makedirs(OUTPUT_ORIG, exist_ok=True)
os.makedirs(OUTPUT_SNV, exist_ok=True)


# ============================================================
# FUNCIONES
# ============================================================

def aplicar_snv(v):
    return (v - np.mean(v)) / np.std(v)


def cargar_muestra(path):
    df = pd.read_excel(path, header=None)

    # Caso 1: dos columnas (lambda, valor)
    if df.shape[1] == 2:
        lambdas = df.iloc[:, 0].to_numpy()
        values  = df.iloc[:, 1].to_numpy()
        return lambdas, values

    # Caso 2: espectro en una fila
    lambdas = np.arange(df.shape[1])
    values  = df.iloc[0, :].to_numpy()
    return lambdas, values


# ============================================================
# PROCESO
# ============================================================

for variedad in VARIEDADES:
    print(f"\nProcesando variedad: {variedad}")

    carpeta = os.path.join(BASE_DIR, variedad)
    archivos = sorted(glob.glob(os.path.join(carpeta, EXT)))

    if not archivos:
        print(f"❌ No se encontraron archivos en: {carpeta}")
        continue

    lambda_ref = None
    matriz = []

    for fpath in archivos:
        lambdas, values = cargar_muestra(fpath)

        if lambda_ref is None:
            lambda_ref = lambdas
        else:
            if len(lambda_ref) != len(lambdas):
                raise ValueError("Las longitudes de onda NO coinciden entre muestras")

        matriz.append(values)

    matriz = np.array(matriz).T     # (nλ x 30 muestras)

    # Crear dataframe original
    df_out = pd.DataFrame(matriz, columns=[f"M{i+1}" for i in range(matriz.shape[1])])
    df_out.insert(0, "lambda", lambda_ref)

    # Guardar dataset original
    out_path = os.path.join(OUTPUT_ORIG, f"{variedad.replace(' ', '_')}_ORIGINAL.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"✔ Dataset original guardado en: {out_path}")

    # Crear dataset SNV (por columna/muestra)
    matriz_snv = np.zeros_like(matriz)

    for i in range(matriz.shape[1]):
        matriz_snv[:, i] = aplicar_snv(matriz[:, i])

    df_snv = pd.DataFrame(matriz_snv, columns=[f"M{i+1}_SNV" for i in range(matriz_snv.shape[1])])
    df_snv.insert(0, "lambda", lambda_ref)

    out_snv = os.path.join(OUTPUT_SNV, f"{variedad.replace(' ', '_')}_SNV.xlsx")
    df_snv.to_excel(out_snv, index=False)
    print(f"✔ Dataset SNV guardado en: {out_snv}")

print("\n🎉 PROCESO FINALIZADO: DATASETS LISTOS")
