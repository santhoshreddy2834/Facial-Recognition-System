import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
import pickle
import uuid
from datetime import datetime

# Load FaceNet model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/facenet_keras.h5'))
face_encoder = load_model(model_path)
detector = MTCNN()

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Paths
faces_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/faces'))
embeddings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/embeddings.pickle'))
attendance_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/attendance.csv'))

# Helper: Preprocess Face
def preprocess_face(img, required_size=(160, 160)):
    image = cv2.resize(img, required_size)
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# Helper: Get Embeddings
def get_embedding(face_img):
    preprocessed = preprocess_face(face_img)
    return face_encoder.predict(preprocessed)[0]

# Helper: Load Embeddings
def load_embeddings():
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            return pickle.load(f)
    return {}

# Helper: Save Embeddings
def save_embeddings(embeddings):
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Register Face
@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"]
    is_criminal = request.form.get("is_criminal", "no")

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = detector.detect_faces(img)
    if not results:
        return jsonify({"error": "No face detected!"}), 400

    x, y, w, h = results[0]['box']
    face = img[y:y + h, x:x + w]

    embedding = get_embedding(face)

    # Save image
    person_dir = os.path.join(faces_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(file_path, face)

    embeddings = load_embeddings()
    embeddings[name] = {
        "embedding": embedding,
        "is_criminal": is_criminal
    }
    save_embeddings(embeddings)

    return jsonify({"message": f"{name} registered successfully."})

# Recognize Face
@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = detector.detect_faces(img)
    if not results:
        return jsonify({"name": "Unknown", "confidence": 0.0})

    x, y, w, h = results[0]['box']
    face = img[y:y + h, x:x + w]

    embedding = get_embedding(face)
    embeddings = load_embeddings()

    recognized_name = "Unknown"
    confidence = 0.0
    is_criminal = "no"

    for name, data in embeddings.items():
        db_embedding = np.array(data["embedding"])
        dist = np.linalg.norm(embedding - db_embedding)
        if dist < 10:
            recognized_name = name
            confidence = float(1 - dist / 10)
            is_criminal = data.get("is_criminal", "no")
            break

    # Log attendance
    with open(attendance_path, "a") as f:
        f.write(f"{recognized_name},{datetime.now()},{is_criminal}\n")

    return jsonify({
        "name": recognized_name,
        "confidence": round(confidence * 100, 2),
        "is_criminal": is_criminal
    })

# Update Embeddings
@app.route("/update_embeddings", methods=["POST"])
def update_embeddings():
    updated = 0
    embeddings = {}

    for person in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            results = detector.detect_faces(img)
            if not results:
                continue

            x, y, w, h = results[0]['box']
            face = img[y:y + h, x:x + w]
            embedding = get_embedding(face)

            embeddings[person] = {
                "embedding": embedding,
                "is_criminal": "no"  # default or could retrieve existing status
            }
            updated += 1
            break  # take only one image per person

    save_embeddings(embeddings)
    return jsonify({"message": f"Updated {updated} embeddings."})

# Get Registered Faces
@app.route("/registered_faces", methods=["GET"])
def registered_faces():
    data = []
    for person in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person)
        if os.path.isdir(person_path):
            image_list = os.listdir(person_path)
            if image_list:
                data.append({
                    "name": person,
                    "image": f"{person}/{image_list[0]}"
                })
    return jsonify(data)

# Get Attendance Logs
@app.route("/attendance_logs", methods=["GET"])
def attendance_logs():
    logs = []
    if os.path.exists(attendance_path):
        with open(attendance_path, "r") as f:
            for line in f.readlines():
                name, time, is_criminal = line.strip().split(",")
                logs.append({
                    "name": name,
                    "time": time,
                    "is_criminal": is_criminal
                })
    return jsonify(logs)

# Run app on port 8000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
