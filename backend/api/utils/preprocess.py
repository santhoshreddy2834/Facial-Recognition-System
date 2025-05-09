from mtcnn import MTCNN
import cv2
import numpy as np

detector = MTCNN()

def extract_face(image):
    results = detector.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        x, y = max(0, x), max(0, y)
        return image[y:y + height, x:x + width]
    return None

def preprocess_face(face_image):
    face = cv2.resize(face_image, (160, 160))
    face = face.astype('float32') / 255.0
    return np.expand_dims(face, axis=0)
