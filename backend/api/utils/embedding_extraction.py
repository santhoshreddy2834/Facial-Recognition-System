import numpy as np

def extract_embeddings(face_img, model):
    # Ensure input shape is (1, 160, 160, 3)
    face_img = np.expand_dims(face_img, axis=0)
    embedding = model.predict(face_img)
    return embedding[0]
