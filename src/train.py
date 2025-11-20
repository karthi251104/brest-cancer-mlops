import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ====================== PATHS ======================


TRAIN_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)\train"
VAL_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)\valid"

# ====================== DATA GENERATORS ======================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ====================== BUILD MODEL ======================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ====================== TRAIN ======================
history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen
)

# ====================== MODEL REGISTRY ======================
# Create models_registry folder if missing
os.makedirs("models_registry", exist_ok=True)

# Timestamp for versioning
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Versioned model path
version_path = f"models_registry/model_{timestamp}.keras"

# Latest model path
latest_path = "models_registry/latest_model.keras"

# Save models
model.save(version_path)
model.save(latest_path)

print("\n==================== MODEL SAVED ====================")
print(f"Versioned model saved to: {version_path}")
print(f"Latest model saved to: {latest_path}")
print("=====================================================\n")