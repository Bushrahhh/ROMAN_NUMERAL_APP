from flask import Flask, render_template, request
import os
import random  # This was likely missing

# Create the app
app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fake labels for testing
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

    # Simulate a fake prediction
    fake_result = random.choice(labels)

    return f"✨ Fake Prediction: {fake_result} (Model not connected yet)"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
