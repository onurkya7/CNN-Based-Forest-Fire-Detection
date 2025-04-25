import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Test image
test_image_path = "F_16.jpg"
target_size = (150, 150)
model = keras.models.load_model("fire_detection_model.h5")
img = image.load_img(test_image_path, target_size=target_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Rescale

# Prediction
y_pred = model.predict(img_array)
prediction = (y_pred > 0.5).astype(int)

class_names = ["fire", "nofire"]
predicted_class = class_names[prediction[0][0]]

print(f'Prediction: {predicted_class}')

# Open the image in high quality
high_quality_img = Image.open(test_image_path)

plt.figure(figsize=(8, 6))  
plt.imshow(high_quality_img)
plt.axis('off')  # Remove axes

# Set title color based on prediction
if predicted_class == 'fire':
    title_color = 'red'
else:
    title_color = 'blue'

# Set the title and its color
plt.title(f'Prediction: {predicted_class}', color=title_color)
plt.show()
