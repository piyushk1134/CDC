import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image #type: ignore

model_path = 'cdm2.keras'
test_dir = 'test_images'
output_dir = 'predicted'
os.makedirs(os.path.join(output_dir, 'cats'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'dogs'), exist_ok=True)
model = tf.keras.models.load_model(model_path)

img_size = (128, 128)
for fname in os.listdir(test_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_path = os.path.join(test_dir, fname)
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    prediction = model.predict(img_array)[0][0]
    label = 'dogs' if prediction > 0.5 else 'cats'
    
    dest_path = os.path.join(output_dir, label, fname)
    shutil.copy(img_path, dest_path)

    print(f"{fname} → {label}")

print("✅ All images classified and sorted.")
