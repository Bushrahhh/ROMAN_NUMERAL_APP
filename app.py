from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Create the app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model.h5')

# Roman numeral labels (index matches class number)
labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess image
    try:
        img = load_img(filepath, color_mode='grayscale', target_size=(28, 28))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)  # reshape to (1, 28, 28, 1)

        # Predict using model
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]

        return render_template('index.html', result=predicted_label, image_file=file.filename)

    except Exception as e:
        return f"Prediction failed: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
