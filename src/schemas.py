"""Contratos Pydantic v2 de la API Fashion MNIST."""

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    """Probabilidad de una clase individual."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Respuesta del endpoint POST /predict.

    Los campos raíz contienen el top-1 y ``resultados`` las 10 clases
    ordenadas de mejor a peor confianza.
    """

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_version: str
    resultados: list[ClassProbability]


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""

    status: str
    model_loaded: bool
