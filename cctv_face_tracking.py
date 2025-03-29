import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN

# Set camera source (0 for USB, "rtsp://your_ip_camera_url" for IP)
camera_source = 0  # Change if using an IP camera

# Load MTCNN for face detection
detector = MTCNN()

# Create folders for logs and unknown faces
os.makedirs("logs", exist_ok=True)
os.makedirs("unknown_faces", exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(camera_source)

# Check if camera opened successfully
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

    # Detect faces
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face["box"]
        face_crop = frame[y:y+h, x:x+w]

        # Verify face using DeepFace
        try:
            result = DeepFace.find(face_crop, db_path="dataset", model_name="Facenet", enforce_detection=False)

            if result and len(result[0]) > 0:
                name = os.path.basename(result[0]["identity"][0]).split(".")[0]
                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces

                # Save unknown face
                face_path = f"unknown_faces/{int(time.time())}.jpg"
                cv2.imwrite(face_path, face_crop)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")

    # Show the CCTV feed
    cv2.imshow("CCTV Face Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

