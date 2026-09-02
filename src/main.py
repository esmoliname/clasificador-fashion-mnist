"""Clasificador de Fashion MNIST con TensorFlow y Keras."""

import random
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.constants import (
    BATCH_SIZE,
    CLASS_COUNT,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HIDDEN_UNITS,
    INPUT_SHAPE,
    LR_REDUCE_PATIENCE,
    LOGS_DIR,
    SEED,
    VALIDATION_SPLIT,
)
from src.evaluate import evaluar_detallado


def cargar_datos() -> (
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
):
    """Carga el conjunto Fashion MNIST desde Keras.

    Returns:
        Tupla ((x_train, y_train), (x_test, y_test)) con arrays numpy.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocesar(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza los píxeles de las imágenes al rango [0, 1] como float32.

    Args:
        x_train: Array de entrenamiento con shape (N, 28, 28), dtype uint8.
        x_test: Array de test con shape (M, 28, 28), dtype uint8.

    Returns:
        Tupla (x_train_norm, x_test_norm) en float32 y rango [0, 1].
    """
    return (
        x_train.astype(np.float32) / 255.0,
        x_test.astype(np.float32) / 255.0,
    )


def construir_modelo(input_shape: tuple[int, int] = INPUT_SHAPE) -> keras.Sequential:
    """Construye el modelo Secuencial de clasificación.

    Arquitectura declarativa: cada hiperparámetro viene de constants.py,
    de modo que la definición vive en UN solo lugar.

    Args:
        input_shape: Tupla (alto, ancho) de las imágenes de entrada.

    Returns:
        Modelo Sequential compilado con adam y sparse_categorical_crossentropy.
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(CLASS_COUNT, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def entrenar(
    model: keras.Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> keras.callbacks.History:
    """Entrena el modelo con callbacks de control de calidad.

    Incluye EarlyStopping, ReduceLROnPlateau y ModelCheckpoint para
    garantizar reproducibilidad y evitar sobreajuste.

    Args:
        model: Modelo Keras compilado.
        x_train: Datos de entrenamiento normalizados.
        y_train: Etiquetas de entrenamiento.

    Returns:
        Objeto History con métricas por época.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=LR_REDUCE_PATIENCE,
            factor=0.5,
            min_lr=1e-5,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(DEFAULT_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
    )
    return history


def exportar_historial(history: keras.callbacks.History) -> Path:
    """Exporta el historial de entrenamiento a logs/training_history.json.

    Escribe loss, accuracy, val_loss y val_accuracy por época.

    Args:
        history: Objeto History devuelto por model.fit.

    Returns:
        Ruta completa del archivo JSON generado.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = LOGS_DIR / "training_history.json"

    historial = {
        "loss": [float(v) for v in history.history["loss"]],
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)

    print(f"Historial exportado a {filepath}")
    return filepath


def evaluar(
    model: keras.Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evalúa el modelo sobre el conjunto de prueba y muestra métricas."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Pérdida en test: {loss:.4f}")
    print(f"Exactitud en test: {accuracy:.4f}")


def main() -> None:
    """Punto de entrada principal del clasificador.

    Establece semillas para reproducibilidad del pipeline completo,
    luego ejecuta: carga → preprocesamiento → construcción → entrenamiento
    → exportación de historial → evaluación básica → evaluación detallada.
    """
    # Semillas para reproducibilidad total del pipeline.
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Pipeline de entrenamiento
    (x_train, y_train), (x_test, y_test) = cargar_datos()
    x_train, x_test = preprocesar(x_train, x_test)
    model = construir_modelo()
    history = entrenar(model, x_train, y_train)

    # Exportar historial y evaluar
    exportar_historial(history)
    evaluar(model, x_test, y_test)
    evaluar_detallado(model, x_test, y_test, CLASS_NAMES)

    # Guardar modelo final (ModelCheckpoint ya guardó el mejor, pero
    # guardamos el resultado final explícitamente).
    model.save(DEFAULT_MODEL_PATH)
    print(f"Modelo guardado en {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
