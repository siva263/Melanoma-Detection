import os
import requests
from flask import Flask, render_template, request, send_from_directory, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

UPLOAD_FOLDER = "uploads/all_class"
STATIC_FOLDER = "static"
MODEL_FILENAME = "VGG19-224-model.06-0.12.hdf5"
MODEL_PATH = os.path.join(STATIC_FOLDER, "VGG19-224-model.06-0.12.hdf5")
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

# Check if the model file exists, if not download it
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to {MODEL_PATH}...")
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    print(f"Downloaded model to {MODEL_PATH}")

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Rest of the code remains the same...


# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load and preprocess the image for prediction
def load_and_preprocess_image():
    test_fldr = 'uploads'
    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_fldr,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    test_generator.reset()
    return test_generator

# Function to classify the uploaded image using the model
def classify(model):
    batch_size = 1
    test_generator = load_and_preprocess_image()
    prob = model.predict(test_generator, steps=len(test_generator) / batch_size)
    labels = {0: 'Just another beauty mark', 1: 'Get that mole checked out'}
    label = labels[1] if prob[0][0] >= 0.5 else labels[0]
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    return label, classified_prob

@app.route("/", methods=['GET'])
def home():
    filelist = os.path.join(UPLOAD_FOLDER, '*.*')
    for filePath in glob.glob(filelist):
        try:
            os.remove(filePath)
        except Exception as e:
            print(f"Error while deleting file {filePath}: {e}")
    return render_template("home.html")

@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            upload_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_image_path)

            label, prob = classify(model)
            prob = round((prob * 100), 2)

            return render_template("classify.html", image_file_name=filename, label=label, prob=prob)
        else:
            return redirect(request.url)

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
