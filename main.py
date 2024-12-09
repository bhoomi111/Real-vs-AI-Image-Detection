from src.preprocess import reduce_dataset, load_images
from src.visualize import display_images, plot_accuracy
from src.model import create_model, augment_data
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Dataset paths
# real_path = "data/train/REAL"
# fake_path = "data/train/FAKE"

# # Dataset reduction
# reduce_dataset(real_path, 'data/train1/REAL_reduced', num_images=5000)
# reduce_dataset(fake_path, 'data/train1/FAKE_reduced', num_images=5000)

# Load reduced datasets
real_images, real_labels = load_images('data/train1/REAL_reduced', label=0)
fake_images, fake_labels = load_images('data/train1/FAKE_reduced', label=1)

# Combine and preprocess data
X = np.concatenate([real_images, fake_images], axis=0) / 255.0  # Normalize to [0, 1]
y = to_categorical(np.concatenate([real_labels, fake_labels], axis=0), 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augment data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training data generator
train_generator = train_datagen.flow(
    X_train, y_train, batch_size=16  # Using in-memory data
)

# Validation data generator (without augmentation)
val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(
    X_test, y_test, batch_size=16  # Using in-memory data
)

# Create the model
model = create_model((64, 64, 3))  # Adjust input shape based on resized images

# Train the model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

# Plot training and validation accuracy
plot_accuracy(history)

# Evaluate the model on the test data
evaluate_model(model, X_test, y_test)

# Save the trained model
model.save('trained_model.h5')
print("Model saved as 'trained_model.h5'")
