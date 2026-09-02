"""API FastAPI del clasificador Fashion MNIST (versión hardened).

Endpoints:
- POST /predict: clasifica una imagen y devuelve top-1 + 10 probabilidades.
- GET /health: healthcheck que nunca lanza excepción.
"""

import io
import logging
import os
import time
from functools import lru_cache

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.constants import (
    ALLOWED_MIME_TYPES,
    CLASS_COUNT,
    CLASS_NAMES,
    CORS_DEFAULT_ORIGINS,
    DEFAULT_MODEL_PATH,
    MAX_FILE_SIZE_BYTES,
    MODEL_VERSION,
)
from src.schemas import ClassProbability, HealthResponse, PredictResponse

# Límite anti "decompression bomb": se aplica antes de abrir cualquier imagen.
Image.MAX_IMAGE_PIXELS = 10_000_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("fashion_mnist")


def _parsear_origenes_cors() -> list[str]:
    """Orígenes CORS permitidos: variable ALLOWED_ORIGINS (coma) o default."""
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return list(CORS_DEFAULT_ORIGINS)


# Carga diferida con caché: la app se importa sin cargar el modelo y los
# tests pueden hacer stub de get_model() sin tocar TensorFlow.
@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    model_path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not os.path.exists(model_path):
        raise RuntimeError(f"No se encontró el modelo en {model_path}")
    return tf.keras.models.load_model(model_path)


def preparar_imagen(image_bytes: bytes) -> np.ndarray:
    """Valida bytes de imagen y produce el tensor (1, 28, 28) float32 en [0, 1]."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as imagen:
            gris = imagen.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
            img_array = np.asarray(gris, dtype=np.float32) / 255.0
            img_array = (
                1.0 - img_array
            )  # Invertir colores: igual que la versión original
            return img_array.reshape((1, 28, 28))
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=400,
            detail="La imagen no es válida o excede el límite de píxeles permitido",
        ) from exc


app = FastAPI(
    title="Fashion MNIST API",
    version=MODEL_VERSION,
    description="API de clasificación de Fashion MNIST con TensorFlow/Keras.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parsear_origenes_cors(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
def predict_image(request: Request, file: UploadFile = File(...)) -> PredictResponse:
    """Clasifica la imagen subida y devuelve top-1 más las 10 probabilidades.

    Endpoint síncrono: FastAPI lo ejecuta en su threadpool, así la inferencia
    no bloquea el event loop.
    """
    image_bytes = file.file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Imagen demasiado grande (máx 5 MB)",
        )

    # Si el cliente no envía content_type, no se rechaza: se intenta abrir igual.
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de contenido no permitido: {file.content_type}",
        )

    tensor = preparar_imagen(image_bytes)

    start = time.perf_counter()
    try:
        predicciones = get_model().predict(tensor, verbose=0)[0]
    except Exception:
        logger.exception("Error durante la inferencia")
        raise HTTPException(status_code=500, detail="Error durante la inferencia")
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

    resultados = [
        ClassProbability(
            class_id=i,
            class_name=CLASS_NAMES[i],
            confidence=float(predicciones[i]),
        )
        for i in range(CLASS_COUNT)
    ]
    resultados.sort(key=lambda resultado: resultado.confidence, reverse=True)
    top = resultados[0]

    client_host = request.client.host if request.client else "desconocido"
    logger.info(
        "prediccion=%s confianza=%.4f latencia_ms=%.2f model_version=%s cliente=%s",
        top.class_name,
        top.confidence,
        latency_ms,
        MODEL_VERSION,
        client_host,
    )

    return PredictResponse(
        class_id=top.class_id,
        class_name=top.class_name,
        confidence=top.confidence,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
        resultados=resultados,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck: reporta si el modelo está cargado sin llegar a explotar."""
    try:
        get_model()
    except Exception:
        logger.warning("Healthcheck: el modelo no está disponible")
        return HealthResponse(status="ok", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True)
