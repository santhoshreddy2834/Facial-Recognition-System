import cv2
import os
from datetime import datetime
from deepface import DeepFace

# Function to save recognized face images
def save_recognized_face(image, name):
    folder = "recognized_faces"
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"{name}_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    print(f"✅ Saved recognized face: {filename}")

# Function to log recognized faces with timestamps
def log_detection(name):
    log_file = "detection_log.txt"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - Recognized: {name}\n")
    print(f"✅ Log updated: {log_file}")

# Open webcam (0 for default webcam, or provide RTSP URL for CCTV)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.find(face_img, db_path="dataset", model_name="VGG-Face", enforce_detection=False)
            if len(result[0]) > 0:
                person_name = os.path.basename(result[0]['identity'][0]).split('.')[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                save_recognized_face(face_img, person_name)
                log_detection(person_name)
        except Exception as e:
            print(f"⚠️ Recognition error: {str(e)}")

    cv2.imshow("Live CCTV Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
