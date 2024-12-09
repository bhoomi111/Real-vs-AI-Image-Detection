import os
from flask import Blueprint, render_template, request, redirect, url_for
from src.model import load_model  # Ensure load_model is defined in src/model.py
from src.preprocess import preprocess_image  # Your existing preprocessing function

main = Blueprint('main', __name__)

# Load the trained model
model = load_model('trained_model.h5')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save uploaded file
        file_path = os.path.join('flask_app/static/uploads', file.filename)
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        is_fake = prediction[0][1] > 0.5  # Assume index 1 corresponds to "fake"

        return render_template('result.html', is_fake=is_fake, file_url=url_for('static', filename=f'uploads/{file.filename}'))
