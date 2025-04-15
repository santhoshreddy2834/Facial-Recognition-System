# backend/face_utils.py

import os
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DATASET_PATH = "backend/dataset"
EMBEDDING_PATH = "backend/embeddings/embeddings.pkl"
ATTENDANCE_LOG = "backend/logs/attendance.csv"

# Load models
detector = MTCNN()
embedder = load_model("backend/models/facenet_keras.h5")  # You must have the model here

def preprocess_face(image):
    face = cv2.resize(image, (160, 160))
    face = face.astype('float32') / 255.0
    return np.expand_dims(face, axis=0)

def get_face(image):
    results = detector.detect_faces(image)
    if results:
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        return image[y:y+h, x:x+w]
    return None

def generate_embedding(face_img):
    face = preprocess_face(face_img)
    embedding = embedder.predict(face)
    return embedding[0]

def save_image(name, image):
    path = os.path.join(DATASET_PATH, name)
    os.makedirs(path, exist_ok=True)
    count = len(os.listdir(path))
    img_path = os.path.join(path, f"{count+1}.jpg")
    cv2.imwrite(img_path, image)
    return img_path

def update_embeddings():
    embeddings = {}
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)
            face = get_face(image)
            if face is not None:
                embedding = generate_embedding(face)
                embeddings[person] = embedding
                break  # One image is enough for now
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(embeddings, f)

def recognize_face(image):
    face = get_face(image)
    if face is None:
        return "Unknown", None
    embedding = generate_embedding(face)
    with open(EMBEDDING_PATH, "rb") as f:
        stored_embeddings = pickle.load(f)
    for name, saved_embedding in stored_embeddings.items():
        similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
        if similarity > 0.6:  # Threshold
            return name, similarity
    return "Unknown", None

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    attendance_data = {
        "Name": name,
        "DateTime": dt_string
    }
    if not os.path.exists(ATTENDANCE_LOG):
        df = pd.DataFrame([attendance_data])
        df.to_csv(ATTENDANCE_LOG, index=False)
    else:
        df = pd.read_csv(ATTENDANCE_LOG)
        df = pd.concat([df, pd.DataFrame([attendance_data])], ignore_index=True)
        df.to_csv(ATTENDANCE_LOG, index=False)

def get_registered_faces():
    return os.listdir(DATASET_PATH)

def get_attendance_log():
    if os.path.exists(ATTENDANCE_LOG):
        return pd.read_csv(ATTENDANCE_LOG)
    return pd.DataFrame(columns=["Name", "DateTime"])
