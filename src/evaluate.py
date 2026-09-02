"""Métricas de evaluación por clase SIN scikit-learn.

Genera un reporte de clasificación estilo sklearn y una matriz de confusión
usando únicamente numpy y stdlib. Objetivo: auditar confusiones críticas
(camisa vs camiseta vs abrigo).
"""

import csv

import numpy as np
from tensorflow import keras

from src.constants import CLASS_COUNT, CLASS_NAMES, LOGS_DIR


def matriz_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_clases: int = CLASS_COUNT,
) -> np.ndarray:
    """Calcula la matriz de confusión usando solo numpy.

    Args:
        y_true: Etiquetas verdaderas (array 1D de enteros).
        y_pred: Etiquetas predichas (array 1D de enteros).
        num_clases: Número total de clases.

    Returns:
        Matriz de confusión con shape (num_clases, num_clases),
        donde [i, j] = cantidad de muestras de clase i predichas como j.
    """
    matriz = np.zeros((num_clases, num_clases), dtype=np.int64)
    for verdadera, predicha in zip(y_true, y_pred):
        matriz[int(verdadera), int(predicha)] += 1
    return matriz


def reporte_clasificacion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombres_clases: tuple[str, ...] = CLASS_NAMES,
) -> str:
    """Genera un reporte de clasificación estilo sklearn con numpy puro.

    Calcula precision, recall y f1-score por clase, más accuracy global,
    macro avg y weighted avg.

    Args:
        y_true: Etiquetas verdaderas (array 1D de enteros).
        y_pred: Etiquetas predichas (array 1D de enteros).
        nombres_clases: Tupla de nombres legibles por clase.

    Returns:
        String con el reporte formateado estilo sklearn.
    """
    num_clases = len(nombres_clases)
    cm = matriz_confusion(y_true, y_pred, num_clases)

    # Métricas por clase
    precisiones = []
    recalls = []
    f1s = []
    supports = []

    for i in range(num_clases):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # falsos positivos: columna i menos diagonal
        fn = cm[i, :].sum() - tp  # falsos negativos: fila i menos diagonal
        support = int(cm[i, :].sum())

        # Precision = tp / (tp + fp), guard contra división por cero
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        # Recall = tp / (tp + fn), guard contra división por cero
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # F1 = 2 * prec * rec / (prec + rec), guard contra división por cero
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisiones.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    # Accuracy global
    total = sum(supports)
    correctos = int(np.trace(cm))
    accuracy = correctos / total if total > 0 else 0.0

    # Macro avg (promedio simple de todas las clases)
    macro_prec = sum(precisiones) / num_clases
    macro_rec = sum(recalls) / num_clases
    macro_f1 = sum(f1s) / num_clases

    # Weighted avg (promedio ponderado por support)
    weighted_prec = (
        sum(p * s for p, s in zip(precisiones, supports)) / total if total > 0 else 0.0
    )
    weighted_rec = (
        sum(r * s for r, s in zip(recalls, supports)) / total if total > 0 else 0.0
    )
    weighted_f1 = (
        sum(f * s for f, s in zip(f1s, supports)) / total if total > 0 else 0.0
    )

    # Formateo estilo sklearn
    max_len = max(len(nombre) for nombre in nombres_clases)
    header = f"{'':>{max_len}}   precision    recall  f1-score   support"
    sep = f"{'':>{max_len}} " + "-" * 45

    lineas = [header, sep]
    for i, nombre in enumerate(nombres_clases):
        lineas.append(
            f"{nombre:>{max_len}}   {precisiones[i]:>9.4f}  {recalls[i]:>9.4f}"
            f"  {f1s[i]:>9.4f}  {supports[i]:>9d}"
        )

    lineas.append(sep)
    lineas.append(
        f"{'accuracy':>{max_len}} " + " " * 14 + f"{accuracy:>9.4f}  {total:>9d}"
    )
    lineas.append(
        f"{'macro avg':>{max_len}} "
        + " " * 14
        + f"{macro_prec:>9.4f}  {macro_rec:>9.4f}  {macro_f1:>9.4f}  {total:>9d}"
    )
    lineas.append(
        f"{'weighted avg':>{max_len}} "
        + " " * 14
        + f"{weighted_prec:>9.4f}  {weighted_rec:>9.4f}  "
        + f"{weighted_f1:>9.4f}  {total:>9d}"
    )

    return "\n".join(lineas)


def evaluar_detallado(
    model: keras.Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    nombres_clases: tuple[str, ...] = CLASS_NAMES,
) -> None:
    """Evalúa el modelo con reporte por clase y matriz de confusión.

    Imprime el reporte de clasificación estilo sklearn y una tabla ASCII
    de la matriz de confusión. Guarda ambos artefactos en logs/.

    Args:
        model: Modelo Keras entrenado.
        x_test: Datos de test normalizados (float32, [0,1]).
        y_test: Etiquetas de test.
        nombres_clases: Nombres legibles de las clases.
    """
    # Predicciones
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Reporte de clasificación
    reporte = reporte_clasificacion(y_test, y_pred, nombres_clases)
    print("\n=== Reporte de Clasificación ===")
    print(reporte)

    # Matriz de confusión
    cm = matriz_confusion(y_test, y_pred, len(nombres_clases))
    max_nombre = max(len(nombre) for nombre in nombres_clases)

    print("\n=== Matriz de Confusión ===")
    # Encabezado con nombres de clase predichos
    encabezado = " " * (max_nombre + 2) + "  ".join(
        f"{nombre[:4]:>5}" for nombre in nombres_clases
    )
    print(encabezado)

    # Filas de la matriz
    for i, nombre in enumerate(nombres_clases):
        fila_vals = "  ".join(f"{cm[i, j]:>5d}" for j in range(len(nombres_clases)))
        print(f"{nombre:>{max_nombre}}  {fila_vals}")

    # Guardar artefactos en logs/
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar reporte como texto
    reporte_path = LOGS_DIR / "classification_report.txt"
    with open(reporte_path, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(f"\nReporte guardado en {reporte_path}")

    # Guardar matriz de confusión como CSV
    cm_path = LOGS_DIR / "confusion_matrix.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Fila 0: encabezado "clase_real/clase_predicha" + nombres predichos
        writer.writerow(["clase_real/clase_predicha"] + list(nombres_clases))
        for i, nombre in enumerate(nombres_clases):
            writer.writerow(
                [nombre] + [int(cm[i, j]) for j in range(len(nombres_clases))]
            )
    print(f"Matriz de confusión guardada en {cm_path}")
