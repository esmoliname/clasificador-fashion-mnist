"""Constantes canónicas del clasificador Fashion MNIST.

Este módulo es la ÚNICA fuente de verdad para nombres de clase y parámetros
de configuración de la API. Cualquier otro módulo debe importar desde aquí
y NO duplicar listas ni valores mágicos.
"""

from pathlib import Path

# ÚNICA fuente de verdad para las 10 clases de Fashion MNIST.
# Casing unificado ("Camiseta/Top") tal como se muestra en la API actual.
CLASS_NAMES: tuple[str, ...] = (
    "Camiseta/Top",
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

# Ruta por defecto del modelo entrenado, relativa a la raíz del repositorio.
DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "modelo.keras"

# Versión del modelo expuesta en la API (semver).
MODEL_VERSION: str = "1.0.0"

# Tamaño máximo de imagen aceptado por POST /predict (5 MB).
MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024

# Tipos MIME permitidos para las imágenes subidas.
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/gif"}
)

# Orígenes CORS por defecto (cliente estático en localhost:10000).
CORS_DEFAULT_ORIGINS: list[str] = [
    "http://localhost:10000",
    "http://127.0.0.1:10000",
]

# Número de clases de Fashion MNIST.
CLASS_COUNT: int = 10

# --- Parámetros de entrenamiento y reproducibilidad ---

# Semilla global para reproducibilidad del pipeline completo.
SEED: int = 42

# Épocas máximas de entrenamiento.
EPOCHS: int = 10

# Tamaño de batch para el entrenamiento.
BATCH_SIZE: int = 32

# Fracción del conjunto de entrenamiento reservada para validación.
VALIDATION_SPLIT: float = 0.2

# Dimensión de entrada de las imágenes (alto, ancho) en píxeles.
INPUT_SHAPE: tuple[int, int] = (28, 28)

# Unidades de la capa Dense oculta (arquitectura canónica).
HIDDEN_UNITS: int = 128

# Tasa de dropout para regularización.
DROPOUT_RATE: float = 0.2

# Paciencia para EarlyStopping (épocas sin mejora antes de detener).
EARLY_STOPPING_PATIENCE: int = 3

# Paciencia para ReduceLROnPlateau (épocas sin mejora antes de reducir LR).
LR_REDUCE_PATIENCE: int = 2

# Directorio de logs de entrenamiento.
LOGS_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
