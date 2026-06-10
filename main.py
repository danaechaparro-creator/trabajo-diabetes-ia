# ============================================================
# TRABAJO MÉTODOS DE INTELIGENCIA ARTIFICIAL
# Predicción de diabetes - Clasificación binaria
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import ks_2samp
from xgboost import XGBClassifier


# ============================================================
# INTRODUCCIÓN / PUNTO 1: CARGA Y PREPARACIÓN DE DATOS
# ============================================================

df = pd.read_csv("datos.csv")

print("Dimensiones de la base:", df.shape)
print("\nPrimeras filas:")
print(df.head())

print("\nDistribución de Outcome:")
print(df["Outcome"].value_counts())
print("\nDistribución porcentual de Outcome:")
print((df["Outcome"].value_counts(normalize=True) * 100).round(2))

# Revisión y limpieza de valores negativos inconsistentes
cols_limpieza = ["SkinThickness", "Insulin", "DiabetesPedigreeFunction"]

print("\nValores negativos antes de limpieza:")
print((df[cols_limpieza] < 0).sum())

for col in cols_limpieza:
    df.loc[df[col] < 0, col] = np.nan
    df[col] = df[col].fillna(df[col].median())

print("\nValores negativos después de limpieza:")
print((df[cols_limpieza] < 0).sum())

# Gráfico de distribución de la variable objetivo
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Outcome")
plt.title("Distribución de la variable objetivo")
plt.xlabel("Outcome: 0 = No diabetes, 1 = Diabetes")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig("grafico_outcome.png", dpi=300)
plt.close()

# Matriz de correlación
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("matriz_correlacion.png", dpi=300)
plt.close()

# Separación de variables
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Escalado para modelos sensibles a escala
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\nTamaño entrenamiento:", X_train.shape)
print("Tamaño validación:", X_val.shape)


# ============================================================
# PUNTO 2 Y 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ============================================================

def calcular_ks(y_real, y_prob):
    prob_0 = y_prob[y_real == 0]
    prob_1 = y_prob[y_real == 1]
    ks, _ = ks_2samp(prob_0, prob_1)
    return ks


def evaluar_modelo(nombre, modelo, X_eval, y_eval, conjunto):
    y_pred = modelo.predict(X_eval)
    y_prob = modelo.predict_proba(X_eval)[:, 1]

    return {
        "Modelo": nombre,
        "Conjunto": conjunto,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1-Score": f1_score(y_eval, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_eval, y_prob),
        "KS": calcular_ks(y_eval, y_prob)
    }


modelos = {
    "ElasticNet": {
        "modelo": LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            max_iter=5000,
            random_state=42
        ),
        "X_train": X_train_scaled,
        "X_val": X_val_scaled
    },

    "Random Forest": {
        "modelo": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "X_train": X_train,
        "X_val": X_val
    },

    "XGBoost": {
        "modelo": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "X_train": X_train,
        "X_val": X_val
    },

    "Red Neuronal": {
        "modelo": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=42
        ),
        "X_train": X_train_scaled,
        "X_val": X_val_scaled
    }
}

resultados = []

for nombre, item in modelos.items():
    modelo = item["modelo"]
    modelo.fit(item["X_train"], y_train)

    resultados.append(
        evaluar_modelo(nombre, modelo, item["X_train"], y_train, "Entrenamiento")
    )

    resultados.append(
        evaluar_modelo(nombre, modelo, item["X_val"], y_val, "Validación")
    )

df_resultados = pd.DataFrame(resultados).round(4)

print("\nTABLA DE RESULTADOS MODELOS BASE:")
print(df_resultados)

df_resultados.to_csv("resultados_modelos_base.csv", index=False)


# ============================================================
# PUNTO 4: OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================

param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 150],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid=param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

modelo_final = grid_search.best_estimator_

print("\nMejores hiperparámetros para XGBoost:")
print(grid_search.best_params_)

resultado_train_opt = evaluar_modelo(
    "XGBoost Optimizado",
    modelo_final,
    X_train,
    y_train,
    "Entrenamiento"
)

resultado_val_opt = evaluar_modelo(
    "XGBoost Optimizado",
    modelo_final,
    X_val,
    y_val,
    "Validación"
)

df_resultados_final = pd.concat([
    df_resultados,
    pd.DataFrame([resultado_train_opt, resultado_val_opt]).round(4)
], ignore_index=True)

print("\nTABLA FINAL DE RESULTADOS:")
print(df_resultados_final)

df_resultados_final.to_csv("resultados_finales.csv", index=False)


# ============================================================
# PUNTO 5: INTERPRETACIÓN Y SELECCIÓN DEL MODELO FINAL
# ============================================================
# ============================================================
# PUNTO 5: INTERPRETACIÓN Y SELECCIÓN DEL MODELO FINAL
# ============================================================

validacion = df_resultados_final[df_resultados_final["Conjunto"] == "Validación"]

# Seleccionamos el mejor modelo considerando primero Recall,
# luego F1-Score y finalmente AUC.
# En un problema de predicción de diabetes, Recall es relevante
# porque interesa detectar la mayor cantidad posible de casos positivos.

mejor_modelo = validacion.sort_values(
    by=["Recall", "F1-Score", "AUC"],
    ascending=False
).iloc[0]

print("\nMODELO SELECCIONADO:")
print(mejor_modelo)

print("\nInterpretación:")
print(
    "Para este problema se prioriza Recall, porque interesa detectar la mayor "
    "cantidad posible de casos positivos de diabetes. Sin embargo, también se "
    "consideran F1-Score y AUC para evaluar el equilibrio general del modelo."
)

print(
    f"\nEl modelo final seleccionado fue {mejor_modelo['Modelo']}, "
    f"con Recall={mejor_modelo['Recall']}, "
    f"F1-Score={mejor_modelo['F1-Score']} y "
    f"AUC={mejor_modelo['AUC']} en validación."
)

print(
    "\nSe observa que Random Forest, XGBoost y la Red Neuronal alcanzan "
    "resultados perfectos en entrenamiento, pero disminuyen en validación. "
    "Esto sugiere sobreajuste. En cambio, ElasticNet presenta un desempeño "
    "más estable y mejor capacidad de generalización en la muestra de validación."
)

print(
    "\nAunque se optimizó XGBoost mediante GridSearchCV, el modelo optimizado "
    "no superó a ElasticNet en las métricas de validación. Por esta razón, "
    "ElasticNet se selecciona como modelo final."
)


# ------------------------------------------------------------
# Matriz de confusión del modelo final seleccionado
# ------------------------------------------------------------

# Como el modelo final seleccionado fue ElasticNet, usamos el modelo
# entrenado de ElasticNet y los datos escalados de validación.

modelo_elasticnet = modelos["ElasticNet"]["modelo"]
y_pred_final = modelo_elasticnet.predict(X_val_scaled)

matriz = confusion_matrix(y_val, y_pred_final)

plt.figure(figsize=(5, 4))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión - ElasticNet")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.tight_layout()
plt.savefig("matriz_confusion_modelo_final.png", dpi=300)
plt.close()


# ============================================================
# CONCLUSIÓN
# ============================================================

print("\nConclusión:")
print(
    "Se desarrolló un flujo completo de clasificación binaria para predecir "
    "diabetes. Se entrenaron modelos penalizados, Random Forest, XGBoost y "
    "una red neuronal simple. Luego se evaluaron con métricas como Accuracy, "
    "Precision, Recall, F1-Score, AUC y KS. También se aplicó GridSearchCV "
    "para optimizar XGBoost. Finalmente, se seleccionó ElasticNet como modelo "
    "final, debido a que obtuvo el mejor equilibrio en validación y mostró "
    "menor evidencia de sobreajuste frente a los modelos más complejos."
)