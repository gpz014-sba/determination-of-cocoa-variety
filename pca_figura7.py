import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Carpetas
DATA_SNV  = r"C:\Users\USER\Desktop\Giancarlo\DATASETS_SNV"
OUTPUT    = r"C:\Users\USER\Desktop\Giancarlo\FIGURAS"

os.makedirs(OUTPUT, exist_ok=True)

# Variedades y colores
VARIEDADES = {
    "Cacao_-_Mazamari":   ("Mazamari",     "red",        "^"),
    "Cacao_Amazonas":     ("Amazonas",     "green",      "o"),
    "CACAO_CHUNCHO":      ("Chuncho",      "black",      "D"),
    "Cacao_Cusco_1":      ("Cusco",        "blue",       "s"),
    "Cacao_Mazamari_2":   ("Trinitario",   "magenta",    "P"),
    "Cacao_Piura":        ("Piura",        "cyan",       "v"),
    "Cacao_San_Martín":   ("San_Martin",   "orange",     "h"),
    "CACAO_SATIPO":       ("Satipo",       "purple",     "*")
}

# ================================
# UNIFICAR TODAS LAS VARIEDADES
# ================================
X_total = []
y_total = []

for folder, (label, color, marker) in VARIEDADES.items():

    file_path = os.path.join(DATA_SNV, f"{folder}_SNV.xlsx")

    df = pd.read_excel(file_path)

    # Eliminar λ=0 si está presente
    df = df.iloc[1:, :]

    # Guardar datos (muestras como vectores)
    X = df.drop(columns=["lambda"]).T.to_numpy()
    
    X_total.append(X)
    y_total += [label] * X.shape[0]

# Concatenar todo
X_all = np.vstack(X_total)

# ================================
# PCA 3 COMPONENTES
# ================================
pca = PCA(n_components=3)
scores = pca.fit_transform(X_all)
pc1, pc2, pc3 = scores[:,0], scores[:,1], scores[:,2]

# ================================
# GRAFICAR 3D
# ================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

start = 0
for folder, (label, color, marker) in VARIEDADES.items():
    file_path = os.path.join(DATA_SNV, f"{folder}_SNV.xlsx")
    df = pd.read_excel(file_path)
    df = df.iloc[1:, :]
    n_samples = df.drop(columns=["lambda"]).shape[1]

    end = start + n_samples
    ax.scatter(pc1[start:end], pc2[start:end], pc3[start:end],
               label=label, s=60, c=color, marker=marker)

    start = end

ax.set_title("PCA de Variedades de cacao", fontsize=14)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

ax.legend(loc="upper right", fontsize=10)
ax.grid(True)

# ================================
# GUARDAR FIGURA 7
# ================================
output_path = os.path.join(OUTPUT, "Figura_7.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("\n✔ FIGURA 7 generada correctamente:")
print(output_path)
    