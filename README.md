# Predicción de Diabetes con Machine Learning

Este proyecto corresponde a un trabajo de la asignatura Métodos de Inteligencia Artificial. El objetivo fue aplicar modelos supervisados de clasificación binaria para predecir si una persona presenta o no diabetes.

## Base de datos

Se utilizó una base de datos con 768 registros y 9 columnas. La variable objetivo es `Outcome`, donde:

* `0`: No presenta diabetes.
* `1`: Presenta diabetes.

Las variables predictoras incluyen información como glucosa, presión arterial, insulina, IMC, edad y antecedentes familiares.

## Modelos utilizados

Se entrenaron y compararon los siguientes modelos:

* Regresión Logística con penalización ElasticNet.
* Random Forest.
* XGBoost.
* Red Neuronal simple.

## Métricas de evaluación

Los modelos fueron evaluados usando métricas de clasificación binaria:

* Accuracy.
* Precision.
* Recall.
* F1-Score.
* AUC.
* KS.

## Optimización

Se aplicó GridSearchCV para optimizar los hiperparámetros de XGBoost. Sin embargo, el modelo optimizado no superó el desempeño de ElasticNet en validación.

## Modelo final

El modelo seleccionado fue ElasticNet, ya que presentó el mejor equilibrio en validación y menor evidencia de sobreajuste frente a modelos más complejos.

## Archivos del repositorio

* `main.py`: código principal del proyecto.
* `datos.csv`: base de datos utilizada.
* `resultados_finales.csv`: tabla final de métricas.
* `resultados_modelos_base.csv`: resultados de los modelos iniciales.
* `grafico_outcome.png`: distribución de la variable objetivo.
* `matriz_correlacion.png`: matriz de correlación.
* `matriz_confusion_modelo_final.png`: matriz de confusión del modelo final.
