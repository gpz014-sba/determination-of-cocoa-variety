import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

# ==============================
# CONFIGURACIÓN
# ==============================
BASE_DIR = r"C:\Users\USER\Desktop\Giancarlo"
SNV_DIR  = os.path.join(BASE_DIR, "DATASETS_SNV")
OUT_DIR = os.path.join(BASE_DIR, "IMAGENES")
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# VARIEDADES REALES (8 clases)
# ==============================
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

# ==============================
# CARGAR DATASET SNV COMPLETO
# ==============================
X_total = []
y_total = []

for variedad in VARIEDADES:
    fname = f"{variedad}_150_SNV.xlsx"
    path = os.path.join(SNV_DIR, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo faltante: {path}")

    df = pd.read_excel(path)

    # Matriz = (150 columnas de reflectancia, 30 filas)
    X = df.drop(columns=["lambda"]).values.T
    y = [variedad] * X.shape[0]

    X_total.append(X)
    y_total += y

X = np.vstack(X_total)
y = np.array(y_total)

# Codificación etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ==============================
# MODELO: GRADIENT BOOSTING
# ==============================
model = GradientBoostingClassifier(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

model.fit(X_train, y_train)

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize="true")

# ==============================
# ROC MULTICLASE (CORREGIDO)
# ==============================
y_score = model.predict_proba(X_test)

# Binarizar SOLO etiquetas del test
y_bin_test = label_binarize(
    y_test,
    classes=np.arange(len(encoder.classes_))
)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ---- (a) Matriz de confusión ----
sns.heatmap(
    cm, annot=True, cmap="Greens", fmt=".2f",
    xticklabels=encoder.classes_, yticklabels=encoder.classes_,
    ax=axes[0]
)

axes[0].set_title("a) Matriz de Confusión (Gradient Boosting)", fontsize=13)
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Real")

# ---- (b) ROC CURVE MULTICLASE ----
for i, clase in enumerate(encoder.classes_):
    fpr, tpr, _ = roc_curve(y_bin_test[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, lw=2, label=f"{clase} (AUC = {roc_auc:.2f})")

axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("b) Curva ROC/AUC Multiclase (Gradient Boosting)", fontsize=13)
axes[1].legend(loc="lower right", fontsize=8)

# Guardar imagen
outpath = os.path.join(OUT_DIR, "Figura_11_GradientBoosting.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.close()

print("✔ FIGURA 11 generada correctamente en:", outpath)
