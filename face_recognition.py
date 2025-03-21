from deepface import DeepFace
import cv2

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to your dataset folder
dataset_path = "dataset"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face region
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        try:
            # Recognize the face using DeepFace
            result = DeepFace.find(img_path=face, db_path=dataset_path, enforce_detection=False)
            
            if len(result) > 0:
                person_name = result[0]['identity'][0].split("\\")[-1]  # Extract filename
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

