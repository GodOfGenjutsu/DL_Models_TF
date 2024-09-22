from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import io

# Load the trained model
model_path = "./mnist_cnn_model.h5"  # Adjust based on your save format
model = tf.keras.models.load_model(model_path)

# Define your Flask app
app = Flask(__name__)
CORS(app)
# Image preprocessing pipeline
def preprocess_image(image):
    # Resize to match model input size and normalize
    image = image.resize((28, 28))
    image = np.array(image) / 255.0  # Scale to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Home route serving the HTML page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    
@app.route("/status", methods=["GET"])  
def running():
    return jsonify({"status": "server is runnig"}), 200
# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    
    # Check if the file has an allowed extension
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}), 400

    image = Image.open(io.BytesIO(image_file.read())).convert("L")  # Convert to grayscale

    # Preprocess the image for model input
    image_array = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)

    return jsonify({"prediction": int(predicted_class[0])})

if __name__ == "__main__":
    server_port = int(os.environ.get("PORT", 3050))
    app.run(debug=True, host="0.0.0.0", port=server_port)
