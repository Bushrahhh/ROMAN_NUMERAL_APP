import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Make sure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = load_model('Romaniansarecool_full_model.h5')
all_labels = ['i','ii','iii','iv','ix','v','vi','vii','viii','x']

def prepare_image(path):
    img = image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        img = prepare_image(filepath)
        prediction = model.predict(img)
        predicted_label = all_labels[np.argmax(prediction)]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
