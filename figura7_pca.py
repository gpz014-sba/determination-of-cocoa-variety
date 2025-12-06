import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# CONFIGURACIÓN
# ==============================
BASE_DIR = r"C:\Users\USER\Desktop\Giancarlo"

ORIG_DIR = os.path.join(BASE_DIR, "DATASETS_ORIGINALES")
SNV_DIR  = os.path.join(BASE_DIR, "DATASETS_SNV")

OUT_DIR = os.path.join(BASE_DIR, "IMAGENES")
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# VARIEDADES
# ==============================
VARIEDADES = {
    "Cacao_San_Martin"   : ("blue",       "o"),
    "Cacao_-_Mazamari"   : ("red",        "^"),
    "Cacao_Piura"        : ("green",      "s"),
    "Cacao_Amazonas"     : ("purple",     "D"),
    "Cacao_Cusco_1"      : ("orange",     "P"),
    "Cacao_Satipo"       : ("gold",       "X"),
    "Cacao_Mazamari_2"   : ("cyan",       "*"),
    "CACAO_CHUNCHO"      : ("black",      "h")
}

# ==============================
# CARGAR MATRIZ (30 × 150)
# ==============================
def cargar_matriz(path):
    df = pd.read_excel(path)
    X = df.drop(columns=["lambda"]).values
    return X.T   # (150 × 30) -> (30 × 150)

# ==============================
# CONSTRUIR MATRIZ GENERAL
# ==============================
def construir_matriz_general(base_dir, tipo):

    X_total = []
    y_total = []

    for variedad in VARIEDADES.keys():

        # tipo = "ORIGINAL" o "SNV"
        filename = f"{variedad}_150_{tipo}.xlsx"
        path = os.path.join(base_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        X = cargar_matriz(path)
        X_total.append(X)
        y_total += [variedad] * X.shape[0]

    return np.vstack(X_total), y_total


# ==============================
# GENERAR PCA
# ==============================
def generar_pca(X, y_labels, filename, titulo):

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    start = 0
    for variedad, (color, marker) in VARIEDADES.items():
        end = start + 30
        Xi = X_pca[start:end]

        ax.scatter(
            Xi[:, 0], Xi[:, 1], Xi[:, 2],
            color=color, marker=marker, s=60,
            label=variedad.replace("_", " ")
        )

        start = end

    ax.set_title(titulo)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend(loc="upper left", fontsize=8)

    outpath = os.path.join(OUT_DIR, filename)
    plt.savefig(outpath, dpi=300)
    plt.close()

    print("✔ Figura generada:", outpath)


# ==============================
# PCA FIGURA 7-A (ORIGINAL)
# ==============================
print("Generando PCA con datos ORIGINALES...")
X_orig, y_orig = construir_matriz_general(ORIG_DIR, "ORIGINAL")
generar_pca(X_orig, y_orig, "PCA_Original.png", "PCA de Variedades de cacao (ORIGINAL)")

# ==============================
# PCA FIGURA 7-B (SNV)
# ==============================
print("Generando PCA con datos SNV...")
X_snv, y_snv = construir_matriz_general(SNV_DIR, "SNV")
generar_pca(X_snv, y_snv, "PCA_SNV.png", "PCA de Variedades de cacao (SNV)")

print("\n🎉 FIGURAS PCA COMPLETADAS CON ÉXITO")
