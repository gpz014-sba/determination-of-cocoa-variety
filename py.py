import os
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# CONFIGURACIÓN DE CARPETAS
# ===========================
BASE_DIR = r"C:\Users\USER\Desktop\Giancarlo"
ORIG_DIR = os.path.join(BASE_DIR, "DATASETS_ORIGINALES")
SNV_DIR  = os.path.join(BASE_DIR, "DATASETS_SNV")
OUT_DIR  = os.path.join(BASE_DIR, "IMAGENES")

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# LISTA DE VARIEDADES
# ===========================
VARIEDADES = [
    "Cacao_-_Mazamari",
    "Cacao_Amazonas",
    "CACAO_CHUNCHO",
    "Cacao_Cusco_1",
    "Cacao_Mazamari_2",
    "Cacao_Piura",
    "Cacao_San_Martin",
    "Cacao_Satipo"
]

# ===========================
# FUNCIÓN PARA CARGAR DATOS
# ===========================
def cargar_dataset(nombre):
    path_orig = os.path.join(ORIG_DIR, f"{nombre}_150_ORIGINAL.xlsx")
    path_snv  = os.path.join(SNV_DIR, f"{nombre}_150_SNV.xlsx")

    df_orig = pd.read_excel(path_orig)
    df_snv  = pd.read_excel(path_snv)

    return df_orig, df_snv


# ===========================
# GRAFICAR SOLO ORIGINAL
# ===========================
def graficar_original(nombre, df):

    lambdas = df["lambda"].values
    muestras = df.drop(columns=["lambda"]).values

    plt.figure(figsize=(8, 5))

    for i in range(muestras.shape[1]):
        plt.plot(lambdas, muestras[:, i], linewidth=1)

    plt.title(f"{nombre} - ORIGINAL")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.grid(True)

    outpath = os.path.join(OUT_DIR, f"{nombre}_ORIGINAL.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✔ Imagen creada: {outpath}")


# ===========================
# GRAFICAR SOLO SNV
# ===========================
def graficar_snv(nombre, df):

    lambdas = df["lambda"].values
    muestras = df.drop(columns=["lambda"]).values

    plt.figure(figsize=(8, 5))

    for i in range(muestras.shape[1]):
        plt.plot(lambdas, muestras[:, i], linewidth=1)

    plt.title(f"{nombre} - SNV")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (SNV)")
    plt.grid(True)

    outpath = os.path.join(OUT_DIR, f"{nombre}_SNV.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✔ Imagen creada: {outpath}")


# ===========================
# PROCESAR TODAS LAS VARIEDADES
# ===========================
for variedad in VARIEDADES:
    df_orig, df_snv = cargar_dataset(variedad)

    graficar_original(variedad, df_orig)
    graficar_snv(variedad, df_snv)

print("\n🎉 Todas las imágenes individuales fueron generadas correctamente.")
