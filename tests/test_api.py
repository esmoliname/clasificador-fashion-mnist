"""Tests de los endpoints HTTP de la API Fashion MNIST.

Cubre health, predict (válida/corrupta/sobredimensionada) y CORS.
"""

from src.constants import CLASS_NAMES, MODEL_VERSION


def _validar_esquema_predict(raiz: dict) -> None:
    """Valida el esquema PredictResponse en la respuesta JSON."""

    class_id = raiz["class_id"]
    assert isinstance(class_id, int)
    assert 0 <= class_id <= 9

    class_name = raiz["class_name"]
    assert class_name in CLASS_NAMES

    confidence = raiz["confidence"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

    latency_ms = raiz["latency_ms"]
    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0

    assert raiz["model_version"] == MODEL_VERSION

    resultados = raiz["resultados"]
    assert isinstance(resultados, list)
    assert len(resultados) == 10
    confianzas = [r["confidence"] for r in resultados]
    clases = [r["confidence"] for r in resultados]
    # Ordenadas de mayor a menor confianza.
    assert clases == sorted(clases, reverse=True)
    # Todas las posiciones entre 0 y 1.
    assert all(0.0 <= c <= 1.0 for c in confianzas)
    # La primera entrada de resultados coincide con la raíz (top-1).
    primer = resultados[0]
    assert primer["class_id"] == class_id
    assert primer["class_name"] == class_name
    assert primer["confidence"] == confidence


def test_health(client):
    """GET /health devuelve 200 con el modelo cargado."""
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["status"] == "ok"
    assert datos["model_loaded"] is True


def test_health_model_ausente(client, monkeypatch):
    """GET /health reporta model_loaded=False si get_model lanza."""
    from src import app

    def _falla():
        raise RuntimeError("modelo no disponible")

    monkeypatch.setattr(app, "get_model", _falla)
    respuesta = client.get("/health")
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["status"] == "ok"
    assert datos["model_loaded"] is False


def test_predict_imagen_valida_gray(client, imagen_gray_png):
    """POST /predict con imagen en escala de grises retorna el esquema completo."""
    respuesta = client.post(
        "/predict",
        files={"file": ("imagen.png", imagen_gray_png, "image/png")},
    )
    assert respuesta.status_code == 200
    _validar_esquema_predict(respuesta.json())


def test_predict_imagen_valida_rgb(client, imagen_rgb_png):
    """POST /predict con imagen RGB confirma que el pipeline rgb→gray funciona."""
    respuesta = client.post(
        "/predict",
        files={"file": ("imagen_rgb.png", imagen_rgb_png, "image/png")},
    )
    assert respuesta.status_code == 200
    _validar_esquema_predict(respuesta.json())


def test_predict_archivo_corrupto(client, archivo_corrupto):
    """POST /predict con bytes inválidos retorna 400/422 con detail en español."""
    respuesta = client.post(
        "/predict",
        files={"file": ("malo.png", archivo_corrupto, "image/png")},
    )
    assert respuesta.status_code in (400, 422)
    datos = respuesta.json()
    # El detail puede venir como string directo o como lista de errores (422).
    detalle = datos.get("detail", "")
    if isinstance(detalle, list):
        texto = " ".join(str(e.get("msg", "")) for e in detalle)
    else:
        texto = str(detalle)
    assert texto.strip()  # detail presente y no vacío


def test_predict_archivo_sobredimensionado(client, archivo_sobredimensionado):
    """POST /predict con archivo > 5 MB retorna 413 con detail indicando 5 MB."""
    respuesta = client.post(
        "/predict",
        files={"file": ("grande.png", archivo_sobredimensionado, "image/png")},
    )
    assert respuesta.status_code == 413
    assert "5 MB" in respuesta.json()["detail"]


def test_cors_origen_permitido(client):
    """GET /health con origen permitido devuelve el header de eco CORS."""
    respuesta = client.get("/health", headers={"Origin": "http://localhost:10000"})
    assert (
        respuesta.headers.get("access-control-allow-origin") == "http://localhost:10000"
    )


def test_cors_origen_rechazado(client):
    """GET /health con origen no permitido NO devuelve header de eco CORS."""
    respuesta = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in respuesta.headers


def test_predict_metodo_no_permitido(client):
    """GET /predict devuelve 405: el endpoint solo acepta POST."""
    respuesta = client.get("/predict")
    assert respuesta.status_code == 405
