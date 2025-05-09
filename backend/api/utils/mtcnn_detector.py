from mtcnn import MTCNN
import cv2

class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_single_face(self, frame):
        result = self.detector.detect_faces(frame)
        if result:
            x, y, w, h = result[0]['box']
            x, y = abs(x), abs(y)
            face = frame[y:y+h, x:x+w]
            return face
        return None

