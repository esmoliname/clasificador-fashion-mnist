"""Fixtures compartidos de la suite de tests de la API Fashion MNIST.

Todas las imágenes son sintéticas en memoria (BytesIO / bytes); no se usa
ningún archivo en disco ni el modelo real "modelo.keras".
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import app


class ModeloFalso:
    """Modelo determinista de prueba: devuelve un softmax fijo (argmax clase 0)."""

    def predict(self, x, verbose=0):  # noqa: ARG002 - firma compatible con keras
        logits = np.array([0.8, 0.1, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005])
        return logits.reshape((1, 10))


@pytest.fixture
def modelo_falso():
    """Instancia del modelo falso para compartir entre fixtures y tests."""
    return ModeloFalso()


@pytest.fixture
def client(monkeypatch, modelo_falso):
    """Cliente TestClient con get_model stubbada por el modelo falso.

    Se hace monkeypatch sobre src.app.get_model para que ninguna llamada
    dependa del modelo real ni de disco.
    """
    monkeypatch.setattr(app, "get_model", lambda: modelo_falso)
    with TestClient(app.app) as c:
        yield c


@pytest.fixture
def imagen_gray_png():
    """Imagen PNG en escala de grises (28x28) con gradiente simple."""
    gradiente = np.linspace(0, 255, 28, dtype=np.float64)
    matriz = np.tile(gradiente, (28, 1)).astype(np.uint8)
    imagen = Image.fromarray(matriz, mode="L")
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def imagen_rgb_png():
    """Imagen PNG RGB (28x28) para validar el pipeline rgb→gray."""
    imagen = Image.new("RGB", (28, 28), color=(120, 120, 120))
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def archivo_corrupto():
    """Bytes que no representan una imagen válida."""
    return b"esto no es una imagen"


@pytest.fixture
def archivo_sobredimensionado():
    """Bytes mayores al límite de 5 MB permitido por la API."""
    return b"\0" * (5 * 1024 * 1024 + 1024)
