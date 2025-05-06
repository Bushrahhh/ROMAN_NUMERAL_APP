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
#     app.run(debug=True)from flask import Flask, request, render_template, jsonify
import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define your upload folder
UPLOAD_FOLDER = 'UPLOADS'  # Make sure this folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('Romaniansarecool_full_model.h5')

# Class labels in the same order as training
all_labels = ['i', 'ii', 'iii', 'iv', 'ix', 'v', 'vi', 'vii', 'viii', 'x']

# Function to load and preprocess one image
def convert_image_to_array(image_path, size=(28, 28)):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize(size)
    return np.array(image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}),400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}),400

    # Save the file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the saved image
    img = image.load_img(filAepath, target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = all_labels[np.argmax(prediction)]
    prediction = model.predict(img_array)
    predicted_label = LABELS[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({
    "prediction": predicted_label,
    "confidence": confidence
        })
 # Send prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)





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
# model = load_model('Romaniansarecool_full_model.h5')

# # Labels in the same order as model was trained
# labels = ['I', 'II', 'III', 'IV','IX', 'V', 'VI', 'VII', 'VIII', 'X']

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

# #     # Predict
#     prediction = model.predict(img_array)
#     predicted_index = np.argmax(prediction)
#     predicted_label = labels[predicted_index]

#     return f"✅ Predicted Roman Numeral: {predicted_label}"

# # Start the app
# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, request, render_template, redirect, url_for
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import os
# import random  # This was likely missing

# app = Flask(__name__)
# UPLOAD_FOLDER = 'my_images'
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
# from flask import Flask, request, render_template, jsonify
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import os
# import random

# app = Flask(__name__)
# UPLOAD_FOLDER = 'my_images'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Create folder if it doesn't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Fake labels
# labels = ['I', 'II', 'III', 'IV', 'IX','V', 'VI', 'VII', 'VIII', 'X']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Simulate a fake prediction
#     fake_result = random.choice(labels)
#     confidence = round(random.uniform(0.8, 1.0), 2)

#     return jsonify({
#         'prediction': fake_result,
#         'confidence': confidence
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, jsonify
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Create the Flask app


# # Folder to save uploaded images
# UPLOAD_FOLDER = 'static/uploaded'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the trained model (replace with your correct model path)
# model = load_model('Rmaniansarecool_full_model.h5')

# # Roman numeral labels (index matches class number)
# labels = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"})

#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     try:
#         # Load and preprocess image
#         img = load_img(filepath, color_mode='grayscale', target_size=(28, 28))
#         img_array = img_to_array(img)
#         img_array = img_array / 255.0  # Normalize the image
#         img_array = np.expand_dims(img_array, axis=0)  # Reshape for the model

#         # Make prediction
#         prediction = model.predict(img_array)
#         predicted_index = np.argmax(prediction)
#         predicted_label = labels[predicted_index]
#         confidence = np.max(prediction)  # Confidence score of the prediction
        
#         # Return the result as JSON
#         return jsonify({"prediction": predicted_label, "confidence": confidence})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# # Run the app
# if _name_ == '_main_':
#     app.run(debug=True)

