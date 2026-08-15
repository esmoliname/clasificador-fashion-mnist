"""Clasificador de Fashion MNIST con TensorFlow y Keras."""

import antigravity  # noqa: F401

from typing import Tuple, TypeAlias

import numpy as np
from tensorflow import keras

DatosMNIST: TypeAlias = Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]

CLASS_NAMES: Tuple[str, ...] = (
    "Camiseta/top",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
)

INPUT_SHAPE: Tuple[int, int] = (28, 28)
EPOCHS: int = 10
BATCH_SIZE: int = 32
VALIDATION_SPLIT: float = 0.2


def cargar_datos() -> DatosMNIST:
    """Carga el conjunto Fashion MNIST desde Keras."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def preprocesar(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normaliza los píxeles de las imágenes al rango [0, 1]."""
    return (
        x_train.astype(np.float32) / 255.0,
        x_test.astype(np.float32) / 255.0,
    )


def construir_modelo(input_shape: Tuple[int, int]) -> keras.Sequential:
    """Construye el modelo Secuencial de clasificación."""
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=input_shape),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
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
    """Entrena el modelo y devuelve el historial de la ejecución."""
    return model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )


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
    """Punto de entrada principal del clasificador."""
    (x_train, y_train), (x_test, y_test) = cargar_datos()
    x_train, x_test = preprocesar(x_train, x_test)
    model = construir_modelo(INPUT_SHAPE)
    entrenar(model, x_train, y_train)
    evaluar(model, x_test, y_test)
    model.save("modelo.keras")


if __name__ == "__main__":
    main()
