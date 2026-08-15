# Clasificador Fashion MNIST

Clasificador de imágenes de prendas de vestir (Fashion MNIST) construido con
TensorFlow y Keras, aplicando estándares profesionales de calidad, seguridad y
Clean Code.

## Arquitectura de la red neuronal

El modelo es una red neuronal secuencial de perceptrones multicapa (MLP):

| Capa | Descripción |
|------|-------------|
| `Flatten` | Aplana cada imagen de 28x28 píxeles en un vector de 784 características. |
| `Dense (128, relu)` | Capa oculta totalmente conectada con activación ReLU. |
| `Dropout (0.2)` | Regularización que desactiva aleatoriamente el 20% de las neuronas para mitigar el sobreajuste. |
| `Dense (10, softmax)` | Capa de salida con una neurona por clase (10 prendas) y activación softmax que produce probabilidades. |

Configuración del entrenamiento:

- **Optimizador**: Adam.
- **Función de pérdida**: `sparse_categorical_crossentropy`.
- **Métrica**: exactitud (`accuracy`).
- **Épocas**: 10, con `validation_split` del 20%.
- **Normalización**: píxeles escalados al rango [0, 1] (validación de entrada).

Las clases clasificadas son: camiseta/top, pantalón, suéter, vestido, abrigo,
sandalia, camisa, zapatilla, bolso y botín.

## Adaptación del OWASP Top 10 para Machine Learning

| OWASP Top 10 | Adaptación en este proyecto |
|--------------|-----------------------------|
| **A01: Validación de entrada** | Las imágenes se validan y normalizan (0-1) en `preprocesar` antes de entrar a la red; el dataset se carga con tipos controlados (`np.ndarray`). |
| **A02: Fallas de autenticación** | No aplica a un modelo sin API expuesta; el código no gestiona credenciales ni datos sensibles. |
| **A06: Componentes vulnerables/obsoletos** | Dependencias fijadas exactamente en `requirements.txt` y auditoría de seguridad con `bandit` en pre-commit y CI. |
| **A07: Fallas de identificación y autenticación (modelos)** | No aplica: el modelo no autentica usuarios; el acceso al entrenamiento es local. |
| **A08: Fallas de integridad** | Los datos no se descargan de fuentes no confiables; `fashion_mnist` proviene del dataset oficial de Keras. |
| **A09: Fallas de registro y monitoreo** | Las métricas de pérdida y exactitud se registran en consola durante evaluación para auditoría del rendimiento. |
| **A10: Envenenamiento de datos / ataques adversarios** | Mitigación parcial: validación y normalización de entrada; el `Dropout` reduce la sensibilidad a perturbaciones pequeñas. |

Además, `.gitignore` excluye datos, pesos y logs (`*.h5`, `saved_model/`,
`logs/`) para no versionar artefactos de entrenamiento.

## Configuración de GitHub Actions

El flujo `.github/workflows/ci.yml` ejecuta la integración continua en cada
push y pull request hacia `main`:

1. `actions/checkout@v4` — descarga el repositorio.
2. `actions/setup-python@v5` — configura Python 3.10.
3. Instalación de dependencias desde `requirements.txt`.
4. **Validación del código y auditoría de seguridad**:
   - `black --check src/` — verifica el formato.
   - `flake8 src/` — analiza estilo y errores de código.
   - `mypy src/` — comprobación estática de tipos.
   - `bandit -r src/` — auditoría de seguridad.

La validación del modelo (entrenamiento y evaluación) se ejecuta localmente
con `python src/main.py`; el pipeline de CI se encarga de la calidad del
código antes de cualquier fusión.

Los mismos chequeos se ejecutan localmente con `pre-commit`:

```bash
pre-commit install
pre-commit run --all-files
```

## Despliegue y ejecución local

### 1. Crear el entorno virtual

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecutar el clasificador

```bash
python src/main.py
```

El script carga el dataset, entrena el modelo y muestra la pérdida y la
exactitud sobre el conjunto de prueba.

### 4. Verificación de calidad y seguridad

```bash
black src/            # Formateo automático
flake8 src/           # Estilo (PEP 8)
mypy src/             # Tipos estáticos
pylint src/main.py    # Análisis estático
bandit -r src/        # Auditoría de seguridad
pre-commit run --all-files  # Todos los hooks
```