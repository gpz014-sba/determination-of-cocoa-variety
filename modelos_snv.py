import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, 
    f1_score, roc_auc_score
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabulate import tabulate

# ==============================
# CONFIGURACIÓN
# ==============================
BASE_DIR = r"C:\Users\USER\Desktop\Giancarlo"
SNV_DIR = os.path.join(BASE_DIR, "DATASETS_SNV")
OUT_DIR = os.path.join(BASE_DIR, "IMAGENES")
OUT_CM = os.path.join(OUT_DIR, "MATRICES_CONFUSION")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_CM, exist_ok=True)

# ==============================
# VARIEDADES
# ==============================
VARIEDADES = [
    "Cacao_-_Mazamari", "Cacao_Amazonas", "CACAO_CHUNCHO",
    "Cacao_Cusco_1", "Cacao_Mazamari_2", "Cacao_Piura",
    "Cacao_San_Martin", "Cacao_Satipo"
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

    X = df.drop(columns=["lambda"]).values.T
    y = [variedad] * X.shape[0]

    X_total.append(X)
    y_total += y

X = np.vstack(X_total)
y = np.array(y_total)

# ==============================
# CODIFICAR ETIQUETAS
# ==============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ==============================
# TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

# ==============================
# ESCALADO (PLS, SVM)
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# FUNCIÓN MATRIZ DE CONFUSIÓN
# ==============================
def plot_cm(y_true, y_pred, classes, titulo, filename):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(titulo)
    outpath = os.path.join(OUT_CM, filename)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# =====================================================
# 1) PLS-DA
# =====================================================
ohe = OneHotEncoder(sparse_output=False)
Y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))

pls = PLSRegression(n_components=10)
pls.fit(X_train_scaled, Y_train_ohe)

Y_pred_test_ohe = pls.predict(X_test_scaled)
y_pred_pls = np.argmax(Y_pred_test_ohe, axis=1)

acc_pls = accuracy_score(y_test, y_pred_pls)
recall_pls = recall_score(y_test, y_pred_pls, average="macro")
precision_pls = precision_score(y_test, y_pred_pls, average="macro")
f1_pls = f1_score(y_test, y_pred_pls, average="macro")
roc_pls = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_pls), average="macro")

# =====================================================
# 2) SVM RBF
# =====================================================
svm_rbf = SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_svm = svm_rbf.predict(X_test_scaled)

acc_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average="macro")
precision_svm = precision_score(y_test, y_pred_svm, average="macro")
f1_svm = f1_score(y_test, y_pred_svm, average="macro")
roc_svm = roc_auc_score(pd.get_dummies(y_test), svm_rbf.predict_proba(X_test_scaled), average="macro", multi_class="ovr")

# =====================================================
# 3) Random Forest
# =====================================================
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=4,
    min_samples_split=4, max_features=0.3, random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf, average="macro")
precision_rf = precision_score(y_test, y_pred_rf, average="macro")
f1_rf = f1_score(y_test, y_pred_rf, average="macro")
roc_rf = roc_auc_score(pd.get_dummies(y_test), rf.predict_proba(X_test), average="macro", multi_class="ovr")

# =====================================================
# 4) XGBoost
# =====================================================
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.9,
    colsample_bytree=0.8, eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb, average="macro")
precision_xgb = precision_score(y_test, y_pred_xgb, average="macro")
f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")
roc_xgb = roc_auc_score(pd.get_dummies(y_test), xgb.predict_proba(X_test), average="macro", multi_class="ovr")

# ==============================
# TABLA FINAL DE COMPARACIÓN
# ==============================
tabla = pd.DataFrame({
    "Modelo": ["PLS-DA", "SVM RBF", "Random Forest", "XGBoost"],
    "Accuracy": [acc_pls, acc_svm, acc_rf, acc_xgb],
    "Recall": [recall_pls, recall_svm, recall_rf, recall_xgb],
    "Precision": [precision_pls, precision_svm, precision_rf, precision_xgb],
    "F1-score": [f1_pls, f1_svm, f1_rf, f1_xgb],
    "ROC AUC": [roc_pls, roc_svm, roc_rf, roc_xgb]
}).round(3)

# ==============================
# MOSTRAR TABLA BONITA EN CONSOLA
# ==============================
print("\n===== TABLA FINAL =====\n")
print(tabulate(tabla, headers="keys", tablefmt="github", showindex=False))

# ==============================
# EXPORTAR TABLA A EXCEL
# ==============================
tabla_path = os.path.join(BASE_DIR, "TABLA_COMPARACION_MODELOS.xlsx")
tabla.to_excel(tabla_path, index=False)

print("\n✔ TABLA COMPARATIVA GUARDADA EN:")
print(tabla_path)

# ==============================
# MOSTRAR TABLA EN LATEX
# ==============================
print("\n===== TABLA EN LaTeX =====\n")
print(tabla.to_latex(index=False))
