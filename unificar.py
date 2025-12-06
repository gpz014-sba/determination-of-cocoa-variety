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
    v = np.array(v, dtype=float)
    return (v - np.mean(v)) / np.std(v)


def cargar_espectro(path):
    """
    Tu archivo real tiene esta estructura:
    Fila 0: Integration Time
    Fila 1: Number of Averages
    Fila 2: Wavelength %
    Fila 3: Timestamp
    Fila 4: Date/Time
    Fila 5: ESPECTRO REAL (INTENSIDADES)
    """
    df = pd.read_excel(path, header=None)

    # Tomar fila 5 (fila 6 visualmente)
    vector = df.iloc[5, :]

    # Convertir a numérico (si hay texto → NaN)
    vector = pd.to_numeric(vector, errors="coerce").to_numpy()

    # Reemplazar NaN por interpolación simple
    # (muchas veces Excel deja 1 o 2 vacíos)
    isnan = np.isnan(vector)
    if isnan.any():
        vector[isnan] = np.interp(np.flatnonzero(isnan), 
                                  np.flatnonzero(~isnan),
                                  vector[~isnan])

    # Generamos longitudes de onda ficticias (0..N-1)
    lambdas = np.arange(len(vector), dtype=float)

    return lambdas, vector


# ============================================================
# PROCESO PRINCIPAL
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

    for f in archivos:
        lambdas, values = cargar_espectro(f)

        if lambda_ref is None:
            lambda_ref = lambdas

        matriz.append(values)

    matriz = np.array(matriz).T  # (n_lambda × n_muestras)

    # Guardar dataset ORIGINAL
    df_orig = pd.DataFrame(matriz, columns=[f"M{i+1}" for i in range(matriz.shape[1])])
    df_orig.insert(0, "lambda", lambda_ref)

    out_orig = os.path.join(OUTPUT_ORIG, f"{variedad.replace(' ', '_')}_ORIGINAL.xlsx")
    df_orig.to_excel(out_orig, index=False)
    print(f"✔ Dataset ORIGINAL guardado: {out_orig}")

    # Guardar dataset SNV
    matriz_snv = np.zeros_like(matriz)

    for i in range(matriz.shape[1]):
        matriz_snv[:, i] = aplicar_snv(matriz[:, i])

    df_snv = pd.DataFrame(matriz_snv, columns=[f"M{i+1}_SNV" for i in range(matriz_snv.shape[1])])
    df_snv.insert(0, "lambda", lambda_ref)

    out_snv = os.path.join(OUTPUT_SNV, f"{variedad.replace(' ', '_')}_SNV.xlsx")
    df_snv.to_excel(out_snv, index=False)
    print(f"✔ Dataset SNV guardado: {out_snv}")

print("\n🎉 TODOS LOS DATASETS UNIFICADOS Y SNV GENERADOS EXITOSAMENTE")
