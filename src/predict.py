import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

IMG_SIZE = 128
MODEL_PATH = "models/cnn_breast_cancer.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

# ------------------------------------------------------------
# Predict function
# ------------------------------------------------------------
def predict_image(img_path):
    print(f"Processing: {img_path}")

    # Load image using cv2
    img = cv2.imread(img_path)

    if img is None:
        print("❌ ERROR: Could not read image. Check path.")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction)

    if prediction > 0.5:
        label = "Malignant (Cancer)"
    else:
        label = "Benign (Normal)"

    print("-------------------------------------")
    print(f"Prediction: {label}")
    print(f"Confidence score: {confidence:.4f}")
    print("-------------------------------------")

# ------------------------------------------------------------
# Run prediction (CHANGE THE PATH BELOW)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Replace this path with your image
    example_path = r"C:\Users\karthikarthika\Downloads\archive (2)\test\malignant\SOB_M_MC-14-16456-400-010.png"
    predict_image(example_path)