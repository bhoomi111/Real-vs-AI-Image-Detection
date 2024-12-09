import os
import random
import shutil
import numpy as np
import cv2

def reduce_dataset(source_dir, dest_dir, num_images=5000):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    all_images = os.listdir(source_dir)
    sampled_images = random.sample(all_images, num_images)

    for img in sampled_images:
        img_path = os.path.join(source_dir, img)
        dest_img_path = os.path.join(dest_dir, img)
        shutil.copy(img_path, dest_img_path)

    print(f"Reduced dataset to {num_images} images and saved to {dest_dir}")

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def load_images(path, label, target_size=(64, 64)):  # Resize images to (64, 64)
    images = []
    labels = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = load_img(img_path, target_size=target_size)  # Resize here
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_image(image_path, target_size=(64, 64)):
    try:
        # Load the image and resize it
        img = load_img(image_path, target_size=target_size)
        # Convert the image to a numpy array
        img_array = img_to_array(img)
        # Normalize pixel values to the range [0, 1]
        img_array = img_array / 255.0
        # Add an additional batch dimension (expected input shape: (1, height, width, channels))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise
