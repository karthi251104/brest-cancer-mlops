import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 16
TEST_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)\test"

# Load Model
model = tf.keras.models.load_model("models/cnn_breast_cancer.h5")

# Test Generator
test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

loss, acc = model.evaluate(test_gen)

print("Test Loss:", loss)
print("Test Accuracy:", acc)

# Save metrics
os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump({"loss": float(loss), "accuracy": float(acc)}, f, indent=2)

print("Saved metrics to reports/metrics.json")