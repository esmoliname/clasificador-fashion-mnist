from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI(title="Fashion MNIST API", version="1.0.0")

# Configurar CORS (Permitir Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = tf.keras.models.load_model("modelo.keras")
nombres_clases = [
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
]


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Procesamiento del Tensor
    img_array = np.array(image)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, [28, 28])
    img_gray = tf.image.rgb_to_grayscale(img_resized) / 255.0
    img_gray = 1.0 - img_gray  # Invertir colores
    img_input = tf.reshape(img_gray, (1, 28, 28))

    # Predicción
    predicciones = model.predict(img_input)[0]
    resultados = [
        {"clase": nombres_clases[i], "probabilidad": float(predicciones[i])}
        for i in range(10)
    ]
    resultados.sort(key=lambda x: x["probabilidad"], reverse=True)

    return {
        "prediccion_principal": resultados[0]["clase"],
        "confianza": resultados[0]["probabilidad"],
        "resultados": resultados,
    }
