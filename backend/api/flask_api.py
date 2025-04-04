import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load criminal database
CRIMINAL_DB_PATH = "criminal_records.csv"

if not os.path.exists(CRIMINAL_DB_PATH):
    df = pd.DataFrame(columns=["ID", "Name", "Image_Path", "Embedding"])
    df.to_csv(CRIMINAL_DB_PATH, index=False)
    print("[SUCCESS] Default `criminal_records.csv` created!")

# Load the existing criminal database
criminal_db = pd.read_csv(CRIMINAL_DB_PATH)
print("[INFO] Criminal database loaded successfully!")

# Function to compute face embedding
def get_face_embedding(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return embedding[0]["embedding"]
    except:
        return None

# Compare detected face with criminal database
def is_criminal(face_embedding):
    if criminal_db.empty:
        return None

    threshold = 0.6  # Adjust as needed
    for _, row in criminal_db.iterrows():
        stored_embedding = np.array(eval(row["Embedding"]))
        distance = np.linalg.norm(face_embedding - stored_embedding)

        if distance < threshold:
            return row["Name"]  # Criminal Matched

    return None  # No match

@app.route("/detect_criminal", methods=["POST"])
def detect_criminal():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image_path = "temp_face.jpg"
    file.save(image_path)

    # Compute embedding for detected face
    face_embedding = get_face_embedding(image_path)
    if face_embedding is None:
        return jsonify({"error": "No face detected"}), 400

    # Check if face matches a criminal
    criminal_name = is_criminal(np.array(face_embedding))

    if criminal_name:
        return jsonify({"status": "Alert", "criminal": criminal_name}), 200
    else:
        return jsonify({"status": "Clear", "message": "No criminal match found"}), 200

if __name__ == "__main__":
    app.run(debug=True)
