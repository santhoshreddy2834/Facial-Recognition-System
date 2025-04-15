# backend/models/update_embeddings.py

import os
import cv2
import numpy as np
import pickle
from keras.models import load_model
from mtcnn import MTCNN

DATASET_DIR = os.path.abspath("../dataset")
MODEL_PATH = os.path.abspath("facenet_keras.h5")
EMBEDDINGS_PATH = os.path.abspath("../embeddings/embeddings.pkl")

model = load_model(MODEL_PATH)
detector = MTCNN()

def preprocess_face(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def get_embedding(face_img):
    preprocessed = preprocess_face(face_img)
    return model.predict(preprocessed)[0]

def update_embeddings():
    embeddings = []

    for filename in os.listdir(DATASET_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = filename.split('_')[0]
            path = os.path.join(DATASET_DIR, filename)

            image = cv2.imread(path)
            faces = detector.detect_faces(image)
            if not faces:
                continue

            x, y, w, h = faces[0]['box']
            face = image[y:y+h, x:x+w]

            embedding = get_embedding(face)
            embeddings.append({"name": name, "embedding": embedding})

    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Updated embeddings for {len(embeddings)} faces.")

# Run script
if __name__ == "__main__":
    update_embeddings()
