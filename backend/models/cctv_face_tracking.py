import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN

def load_known_faces(dataset_path):
    known_faces = {}
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):  # Ensure it's a folder
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                known_faces[person] = img_path  # Map name to image
    return known_faces

# Set camera source (0 for USB, "rtsp://your_ip_camera_url" for IP)
camera_source = 0  # Change if using an IP camera

detector = MTCNN()
os.makedirs("logs", exist_ok=True)
os.makedirs("unknown_faces", exist_ok=True)

dataset_path = "dataset"
known_faces = load_known_faces(dataset_path)

cap = cv2.VideoCapture(camera_source)
if not cap.isOpened():
    print("[ERROR] Cannot open camera. Check connection.")
    exit()

print("[INFO] CCTV Face Tracking System Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame. Reconnecting...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(camera_source)
        continue

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face["box"]
        face_crop = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.find(face_crop, db_path=dataset_path, model_name="Facenet", enforce_detection=False)

            if result and len(result[0]) > 0:
                matched_identity = result[0]["identity"][0]
                for name, img_path in known_faces.items():
                    if img_path in matched_identity:
                        detected_name = name
                        break
                else:
                    detected_name = "Unknown"
                    color = (0, 0, 255)
            else:
                detected_name = "Unknown"
                color = (0, 0, 255)
                face_path = f"unknown_faces/{int(time.time())}.jpg"
                cv2.imwrite(face_path, face_crop)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, detected_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")

    cv2.imshow("CCTV Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
