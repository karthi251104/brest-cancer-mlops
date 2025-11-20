# deploy/app.py
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI(title="Breast Cancer Classifier")   # <- MUST be named `app`

# load model (ensure this path exists)
MODEL_PATH = "models_registry/latest_model.keras"
# lazy load to surface clearer errors
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # If loading fails, keep model as None but allow server to start so you can see the error in logs
    model = None
    print(f"Warning: failed to load model from {MODEL_PATH}: {e}")

IMG_SIZE = (128, 128)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            return JSONResponse({"error": "Model not loaded on server"}, status_code=500)

        image_bytes = await file.read()
        processed = preprocess_image(image_bytes)
        prediction = float(model.predict(processed)[0][0])

        label = "Malignant" if prediction > 0.5 else "Benign"

        return JSONResponse(content={
            "prediction": label,
            "confidence": prediction
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)