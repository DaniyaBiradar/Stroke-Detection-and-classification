import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Step 1: Data Preparation
zip_file_path = "path1"
extract_folder = "path2"

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Step 2: Data Preprocessing
# Define image dimensions
img_height, img_width = 150, 150

# Create data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images from directory
train_generator = datagen.flow_from_directory(
    extract_folder,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',  # Since we have two classes
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    extract_folder,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Step 3: Model Building
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 4: Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
# Save the trained model
model.save('modelname.h5')

# Step 5: Model Evaluation
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print('\nTest accuracy:', test_acc)


