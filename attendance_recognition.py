import cv2
import pandas as pd
import os
from datetime import datetime
from deepface import DeepFace

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to dataset folder
dataset_path = "dataset"

# Path to attendance CSV file
attendance_file = "attendance.csv"

# Initialize attendance file if not exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

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
            # Recognize the face
            result = DeepFace.find(img_path=face, db_path=dataset_path, enforce_detection=False)

            if len(result) > 0:
                person_name = result[0]['identity'][0].split("\\")[-1]  # Extract filename
                person_name = os.path.splitext(person_name)[0]  # Remove file extension

                # Get current date and time
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                # Read existing attendance records
                df = pd.read_csv(attendance_file)

                # Check if the person is already marked for today
                if not ((df["Name"] == person_name) & (df["Date"] == date)).any():
                    # Append new attendance record
                    new_data = pd.DataFrame([[person_name, date, time]], columns=["Name", "Date", "Time"])
                    df = pd.concat([df, new_data], ignore_index=True)
                    df.to_csv(attendance_file, index=False)
                    print(f"✅ Attendance marked for {person_name} at {time}")

                # Display recognized name on screen
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            print("Error:", e)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
