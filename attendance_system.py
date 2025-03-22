import cv2
import pandas as pd
import os
from datetime import datetime
from deepface import DeepFace

# Reference image for recognition
reference_image = "dataset/sample1.jpg"
person_name = "User"  # Change this to the actual person's name

# Attendance file
attendance_file = "attendance.csv"

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load or create the attendance file
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

print("🎥 Starting webcam... Press 'r' to recognize and mark attendance.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't capture video frame!")
        break

    cv2.imshow("Webcam Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'r' to recognize the face and mark attendance
    if key == ord('r'):
        try:
            print("🔍 Recognizing face...")
            result = DeepFace.verify(img1_path=reference_image, img2_path=frame, model_name="Facenet")

            if result["verified"]:
                print(f"✅ {person_name} recognized! Marking attendance...")
                
                # Read existing attendance data
                df = pd.read_csv(attendance_file)

                # Check if the person is already marked
                if person_name not in df["Name"].values:
                    # Mark attendance
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = df.append({"Name": person_name, "Time": now}, ignore_index=True)
                    df.to_csv(attendance_file, index=False)
                    print(f"✅ Attendance marked for {person_name} at {now}.")
                else:
                    print("⚠️ Attendance already marked for today.")

            else:
                print("❌ No Match! Face not recognized.")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
