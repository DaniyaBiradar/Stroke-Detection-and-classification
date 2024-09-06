import numpy as np
import serial
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model('modelname.h5')

# Function to preprocess the input image
def preprocess_image(img_path, target_size=(150, 150)):
    # Open the image
    img = Image.open(img_path)
    # Convert to RGB mode (remove alpha channel if it exists)
    img = img.convert("RGB")
    # Resize the image to match the target size expected by the model
    img = img.resize(target_size)
    # Convert the image to a NumPy array and normalize pixel values
    img_array = image.img_to_array(img) / 255.0
    # Expand the dimensions to create a batch with a single image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image_class(img_path, model):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction[0][0] >= 0.5:
        return "Hemorrhagic"
    else:
        return "Ischemic"

# Example usage
image_path = image_path = 'path.png'
prediction = predict_image_class(image_path, model)
print("Prediction:", prediction)

arduino_port = 'COM3'  # Adjust the port according to your Arduino's port
arduino = serial.Serial(arduino_port, 9600)
time.sleep(2)  # Allow time for the Arduino to initialize

# Send prediction result to Arduino
image_path = 'path.png'  # Adjust the image path
prediction = predict_image_class(image_path, model)
arduino.write(prediction.encode())

# Close serial connection
arduino.close()
