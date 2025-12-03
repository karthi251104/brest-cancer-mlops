import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# ========================
# FASTAPI APP
# ========================
app = FastAPI(title="Breast Cancer Classifier")

# ========================
# ROOT ENDPOINT (RENDER FIX)
# ========================
@app.get("/")
def home():
    return {
        "message": "Breast Cancer Classifier API is running",
        "docs_url": "/docs"
    }

# ========================
# LOAD MODEL
# ========================
MODEL_PATH = "models_registry/latest_model.keras"

model = None

try:
    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully ✅")
except Exception as e:
    print(f"❌ Model load failed: {e}")

# ========================
# IMAGE PREPROCESSING
# ========================
IMG_SIZE = (128, 128)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ========================
# PREDICTION ENDPOINT
# ========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse(status_code=500, content={
                "error": "Model not loaded on server"
            })

        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        prediction = model.predict(image)[0][0]
        label = "Malignant" if prediction > 0.5 else "Benign"

        return {
            "prediction": label,
            "confidence": float(prediction)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e)
        })
