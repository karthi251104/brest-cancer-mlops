import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)"
TRAIN_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)\train"
VAL_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)\valid"
TEST_DIR = r"C:\Users\karthikarthika\Downloads\archive (2)\test"

# Create folders if not exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Classes in your dataset
CLASSES = ["benign", "malignant"]

for cls in CLASSES:
    raw_class_dir = os.path.join(RAW_DIR, cls)
    images = os.listdir(raw_class_dir)

    # 80/10/10 split
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Create class folders
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    # Copy images
    for img in train_imgs:
        shutil.copy(os.path.join(raw_class_dir, img), os.path.join(TRAIN_DIR, cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(raw_class_dir, img), os.path.join(VAL_DIR, cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(raw_class_dir, img), os.path.join(TEST_DIR, cls, img))

print("Dataset split completed successfully!")