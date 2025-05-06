# from flask import Flask, render_template, request
# import os
# import random  # This was likely missing

# # Create the app
# app = Flask(__name__)

# # Folder to save uploaded images
# UPLOAD_FOLDER = 'static/uploaded'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Fake labels for testing
# labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file uploaded."

#     file = request.files['file']
#     if file.filename == '':
#         return "No file selected."

#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Simulate a fake prediction
#     fake_result = random.choice(labels)

#     return f"✨ Fake Prediction: {fake_result} (Model not connected yet)"

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import os
# import random
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Create the app
# app = Flask(__name__)

# # Folder to save uploaded images
# UPLOAD_FOLDER = 'static/uploaded'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load your trained model
# model = load_model('Romaniansarecool_full_model.h5')

# # Define your label order (this must match how your model was trained)
# all_labels = ['I', 'II', 'III', 'IV', 'IX', 'V', 'VI', 'VII', 'VIII', 'X']

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file uploaded."

#     file = request.files['file']
#     if file.filename == '':
#         return "No file selected."

#     # Save the uploaded file
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Load and preprocess the image
#     img = image.load_img(filepath, target_size=(28, 28))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize
#     prediction = model.predict(img_array)
#     # Predict using the model
#     predicted_label = all_labels[np.argmax(prediction)]


#     return f"✅ Predicted Roman Numeral: {predicted_label}"

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Initialize the app
# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'static/uploaded'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the trained model
# model = load_model('model.h5')

# # Labels in the same order as model was trained
# labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# # Routes
# @app.route('/')
# def index():
#     return '''
#     <h2>Upload Roman Numeral Image</h2>
#     <form method="post" action="/predict" enctype="multipart/form-data">
#         <input type="file" name="file" required>
#         <input type="submit" value="Predict">
#     </form>
#     '''

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "❌ No file uploaded."

#     file = request.files['file']
#     if file.filename == '':
#         return "❌ No file selected."

#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Process the image
#     img = image.load_img(filepath, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     # Predict
#     prediction = model.predict(img_array)
#     predicted_index = np.argmax(prediction)
#     predicted_label = labels[predicted_index]

#     return f"✅ Predicted Roman Numeral: {predicted_label}"

# # Start the app
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import random  # This was likely missing

app = Flask(__name__)
UPLOAD_FOLDER = 'my_images'
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
