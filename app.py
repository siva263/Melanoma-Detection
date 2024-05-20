import os
import glob
import requests
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads/all_class"
STATIC_FOLDER = "static"
MODEL_FILENAME = "VGG19-224-model.06-0.12.hdf5"
MODEL_PATH = os.path.join(STATIC_FOLDER, MODEL_FILENAME)
GOOGLE_DRIVE_FILE_ID = '1GcI419Ev7zwrSpWnEHVg4xDYI4kyuj-b'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
IMAGE_SIZE = 224

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def download_file_from_google_drive(file_id, dest_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest_path)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, dest_path):
    CHUNK_SIZE = 32768

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to {MODEL_PATH}...")
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            upload_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_image_path)

            # Add code here to classify the uploaded image using the `model`

    return render_template("home.html")

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
