from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define class labels
class_labels = ['Early Blight', 'Late Blight', 'Healthy']

def predict_disease(image_path):
    img = Image.open(image_path)  
    img_array = np.expand_dims(np.array(img), axis=0)  
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_labels[predicted_class], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Ensure 'uploads' directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        prediction, confidence = predict_disease(file_path)
        return render_template('result.html', prediction=prediction, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
