"""Tests de las funciones de preprocesamiento (sin red neuronal).

No dependen de get_model ni del modelo real: solo tensores numpy.
"""

import numpy as np
import pytest
from fastapi import HTTPException

from src import app, main


def test_preparar_imagen_dimensiones(imagen_gray_png):
    """preparar_imagen devuelve un tensor (1, 28, 28) en float32."""
    tensor = app.preparar_imagen(imagen_gray_png)
    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == np.float32


def test_preparar_imagen_normalizacion(imagen_gray_png):
    """preparar_imagen normaliza los valores al rango [0, 1] estricto."""
    tensor = app.preparar_imagen(imagen_gray_png)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocesar_rango_y_forma():
    """preprocesar normaliza a float32 en [0, 1] conservando la forma."""
    rng = np.random.RandomState(42)  # determinista
    x_train = rng.randint(0, 256, size=(4, 28, 28)).astype(np.float64)
    x_test = rng.randint(0, 256, size=(4, 28, 28)).astype(np.float64)
    x_train_norm, x_test_norm = main.preprocesar(x_train, x_test)
    for ary, original in ((x_train_norm, x_train), (x_test_norm, x_test)):
        assert ary.dtype == np.float32
        assert ary.shape == original.shape == (4, 28, 28)
        assert ary.min() >= 0.0
        assert ary.max() <= 1.0


def test_preparar_imagen_corrupta():
    """preparar_imagen con bytes inválidos lanza HTTPException 400."""
    with pytest.raises(HTTPException) as excinfo:
        app.preparar_imagen(b"basura")
    assert excinfo.value.status_code == 400
