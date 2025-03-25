import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
import time
import os

detector = MTCNN()
cap = cv2.VideoCapture(0)  # Change to CCTV stream URL if needed

KNOWN_FACES_DIR = "dataset"  # Directory containing known faces
log_file = "detection_log.txt"

# Load known face encodings
def load_known_faces():
    known_faces = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        try:
            encoding = DeepFace.represent(path, model_name="Facenet")[0]['embedding']
            known_faces[filename.split('.')[0]] = np.array(encoding)  # Store as NumPy array
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print("Known faces loaded:", list(known_faces.keys()))  # Debugging output
    return known_faces

known_faces = load_known_faces()

unknown_count = 1  # Counter for unknown faces
def recognize_face(face_img):
    global unknown_count
    try:
        face_embedding = DeepFace.represent(face_img, model_name="Facenet")[0]['embedding']
        face_embedding = np.array(face_embedding)  # Convert to NumPy array
        
        best_match = None
        best_distance = float("inf")

        for name, known_encoding in known_faces.items():
            similarity = np.linalg.norm(face_embedding - known_encoding)
            if similarity < best_distance:  # Find the closest match
                best_distance = similarity
                best_match = name
        
        # Adjusted threshold (Facenet typically uses 0.6 - 1.0 for good matches)
        if best_distance < 1.0:
            return best_match, best_distance
        else:
            unknown_name = f"Unknown_{unknown_count}"
            unknown_count += 1
            return unknown_name, None
    except Exception as e:
        print("Face recognition error:", e)
    
    return "Unknown", None

def log_detection(name):
    with open(log_file, "a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {name}\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    for detection in detections:
        x, y, w, h = detection['box']
        face = frame[max(0, y):max(0, y+h), max(0, x):max(0, x+w)]  # Ensure valid cropping

        name, similarity = recognize_face(face)
        
        if "Unknown" in name:
            face_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(face_path, face)
            encoding = DeepFace.represent(face_path, model_name="Facenet")[0]['embedding']
            known_faces[name] = np.array(encoding)
            print(f"New face saved: {name}")
        
        log_detection(name)

        color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        if similarity is not None:
            text = f"{name} ({similarity:.2f})"
        else:
            text = name
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("CCTV Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
