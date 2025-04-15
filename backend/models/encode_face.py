# backend/models/encode_face.py

import os
import cv2
import numpy as np
import pickle
from keras.models import load_model
from mtcnn import MTCNN

# Paths
MODEL_PATH = os.path.abspath("facenet_keras.h5")
EMBEDDINGS_PATH = os.path.abspath("../embeddings/embeddings.pkl")

# Load model and detector
model = load_model(MODEL_PATH)
detector = MTCNN()

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def get_embedding(face_img):
    preprocessed = preprocess_face(face_img)
    return model.predict(preprocessed)[0]

def save_embedding(name, face_img):
    embedding = get_embedding(face_img)

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
    else:
        data = []

    data.append({"name": name, "embedding": embedding})

    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

    return True

# Example Usage
if __name__ == "__main__":
    img_path = input("Enter path to image: ")
    name = input("Enter name: ")

    image = cv2.imread(img_path)
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        face_img = image[y:y+h, x:x+w]
        save_embedding(name, face_img)
        print(f"Embedding saved for {name}")
    else:
        print("No face detected.")
