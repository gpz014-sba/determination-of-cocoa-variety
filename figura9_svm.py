import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==============================
# CONFIGURACIÓN
# ==============================
BASE_DIR = r"C:\Users\USER\Desktop\Giancarlo"
SNV_DIR  = os.path.join(BASE_DIR, "DATASETS_SNV")
OUT_DIR = os.path.join(BASE_DIR, "IMAGENES")
os.makedirs(OUT_DIR, exist_ok=True)

VARIEDADES = [
    "Cacao_San_Martin",
    "Cacao_-_Mazamari",
    "Cacao_Piura",
    "Cacao_Amazonas",
    "Cacao_Cusco_1",
    "Cacao_Satipo",
    "Cacao_Mazamari_2",
    "CACAO_CHUNCHO"
]

# ==============================
# CARGAR DATASET SNV COMPLETO
# ==============================
X_total = []
y_total = []

for variedad in VARIEDADES:
    file_path = os.path.join(SNV_DIR, f"{variedad}_150_SNV.xlsx")
    df = pd.read_excel(file_path)

    X = df.drop(columns=["lambda"]).values.T  # (30 × 150)
    y = [variedad] * X.shape[0]

    X_total.append(X)
    y_total += y

X = np.vstack(X_total)
y = np.array(y_total)

# Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ==============================
# ENTRENAMIENTO SVM
# ==============================
model = SVC(kernel='rbf', probability=True, gamma='scale', C=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

model.fit(X_train, y_train)

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')

# ==============================
# CURVA DE APRENDIZAJE
# ==============================
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y_encoded, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
test_mean  = np.mean(test_scores, axis=1)
test_std   = np.std(test_scores, axis=1)

# ==============================
# FIGURA 9
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- (a) MATRIZ DE CONFUSIÓN ----
sns.heatmap(
    cm, annot=True, cmap="Greens",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    ax=axes[0], fmt=".2f"
)
axes[0].set_title("a) Matriz de Confusión (SVM)")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Real")

# ---- (b) CURVA DE APRENDIZAJE ----
axes[1].plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
axes[1].plot(train_sizes, test_mean,  'o-', color="green", label="Validation score")

axes[1].fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, color="blue", alpha=0.2)
axes[1].fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, color="green", alpha=0.2)

axes[1].set_title("b) Curva de Aprendizaje (SVM)")
axes[1].set_xlabel("Training Size")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

# Guardar figura
outpath = os.path.join(OUT_DIR, "Figura_9_SVM.png")
plt.savefig(outpath, dpi=300)
plt.close()

print("✔ Figura 9 generada correctamente en:", outpath)
