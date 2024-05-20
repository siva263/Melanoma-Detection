import os
import glob
import requests
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads/all_class"
STATIC_FOLDER = "static"
MODEL_FILENAME = "model.06-0.12.hdf5"
MODEL_PATH = os.path.join(STATIC_FOLDER, MODEL_FILENAME)
GOOGLE_DRIVE_FILE_ID = 'YOUR_FILE_ID'  # Replace with your Google Drive file ID

IMAGE_SIZE = 224

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
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Ensure the model is downloaded and loaded
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to {MODEL_PATH}...")
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)

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

# Predict & classify image
def classify(model):
    batch_size = 1
    test_generator = load_and_preprocess_image()
    prob = model.predict(test_generator, steps=len(test_generator) / batch_size)
    labels = {0: 'Just another beauty mark', 1: 'Get that mole checked out'}
    label = labels[1] if prob[0][0] >= 0.5 else labels[0]
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    return label, classified_prob

# Home page
@app.route("/", methods=['GET'])
def home():
    filelist = glob.glob("uploads/all_class/*.*")
    for filePath in filelist:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file")
    return render_template("home.html")

@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model)
        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
