import cv2
import os
import pandas as pd
from datetime import datetime

# Folders & Files
dataset_folder = "dataset"
recognized_folder = "recognized_faces"
attendance_file = "attendance.csv"
log_file = "detection_log.txt"

# Ensure required folders exist
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(recognized_folder, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

def capture_face():
    """Captures a live face image and saves it to the dataset."""
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(dataset_folder, f"captured_{timestamp}.jpg")
    
    cv2.imwrite(filename, frame)
    print(f"✅ Live face captured: {filename}")
    return filename

def update_attendance(name):
    """Updates the attendance CSV file with recognized face details."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Timestamp"])
        df.to_csv(attendance_file, index=False)  # Create file if not exists

    df = pd.read_csv(attendance_file)
    new_entry = pd.DataFrame([[name, timestamp]], columns=["Name", "Timestamp"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    
    print(f"✅ Attendance updated: {attendance_file}")

def log_detection(name):
    """Logs recognized faces with timestamps."""
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - Recognized: {name}\n")
    print(f"✅ Log updated: {log_file}")

# MAIN PROCESS
if __name__ == "__main__":
    print("📷 Capturing face...")
    image_path = capture_face()

    if image_path:
        # Simulating recognition (Replace this with actual face recognition logic)
        recognized_name = "Person_Identified"  # Example recognized name

        if recognized_name:
            update_attendance(recognized_name)
            log_detection(recognized_name)
            print(f"✅ Face recognized as {recognized_name}")
        else:
            print("❌ No match found in the dataset.")
    
    cap.release()
    cv2.destroyAllWindows()

