# backend/models/detect_face.py

import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

detector = MTCNN()

def detect_and_display(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    path = input("Enter image path: ")
    detect_and_display(path)
