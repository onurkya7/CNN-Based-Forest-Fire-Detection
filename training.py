import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import shutil
from collections import Counter

data_dir = "Training"
classes = ["fire", "nofire"]

# Clean corrupted or unsupported image files
for cls in classes:
    folder_path = os.path.join(data_dir, cls)
    for filename in glob.glob(os.path.join(folder_path, "*.*")):
        try:
            with Image.open(filename) as img:
                rgb_img = img.convert("RGB")
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                rgb_img.save(new_filename, "JPEG")
                if filename != new_filename:
                    os.remove(filename)
        except Exception as e:
            print(f"Removing invalid file: {filename} - Error: {e}")
            os.remove(filename)

target_size = (150, 150)
batch_size = 64

image_paths = {cls: glob.glob(os.path.join(data_dir, cls, "*.jpg")) for cls in classes}
min_samples = min(len(image_paths["fire"]), len(image_paths["nofire"]))

# Create new training and validation directories
train_dir = "Training_Processed"
val_dir = "Validation_Processed"
shutil.rmtree(train_dir, ignore_errors=True)
shutil.rmtree(val_dir, ignore_errors=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    
    selected_images = image_paths[cls][:min_samples]
    train_split = int(0.8 * min_samples)
    
    for i, img_path in enumerate(selected_images):
        if i < train_split:
            shutil.copy(img_path, os.path.join(train_dir, cls))
        else:
            shutil.copy(img_path, os.path.join(val_dir, cls))

# Create training and validation data generators
datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size, class_mode='binary')
val_generator = datagen.flow_from_directory(val_dir, target_size=target_size, batch_size=batch_size, class_mode='binary')

train_counts = Counter(train_generator.classes)
val_counts = Counter(val_generator.classes)
print(f"Training set: Fire: {train_counts[0]}, NoFire: {train_counts[1]}")
print(f"Validation set: Fire: {val_counts[0]}, NoFire: {val_counts[1]}")

# CNN model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)

# Save the model
model.save("fire_detection_model.h5")

print("Model trained and saved successfully.")
