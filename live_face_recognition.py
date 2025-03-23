import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Ensure dataset and logs exist
DATASET_FOLDER = "dataset"
RECOGNIZED_FOLDER = "recognized_faces"
LOG_FILE = "detection_log.txt"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(RECOGNIZED_FOLDER, exist_ok=True)

# Load attendance file if exists, else create new
def initialize_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Timestamp"])
        df.to_csv(ATTENDANCE_FILE, index=False)

# Function to capture image and get name
def capture_face():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Press 's' to save", frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            name = input("Enter person's name: ")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{DATASET_FOLDER}/{name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return filename, name
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None, None

# Function to recognize the face
def recognize_face(image_path, name):
    dataset_images = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER)]
    
    for img in dataset_images:
        try:
            result = DeepFace.verify(image_path, img, model_name="VGG-Face", enforce_detection=False)
            if result['verified']:
                print(f"✅ Match Found! The person is {name}")
                save_recognized_face(image_path, name)
                update_log(name)
                update_attendance(name)
                return
        except:
            continue
    print("❌ No Match Found! Saving as a new entry.")

# Save recognized face image
def save_recognized_face(image_path, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECOGNIZED_FOLDER, f"{name}_{timestamp}.jpg")
    image = cv2.imread(image_path)
    cv2.imwrite(filename, image)
    print(f"✅ Saved recognized face: {filename}")

# Log recognized person
def update_log(name):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} - Recognized: {name}\n")
    print(f"✅ Log updated for {name}")

# Update attendance
def update_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [name, timestamp]
    df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"✅ Attendance updated for {name}")

if __name__ == "__main__":
    initialize_attendance()
    img_path, person_name = capture_face()
    if img_path and person_name:
        recognize_face(img_path, person_name)
