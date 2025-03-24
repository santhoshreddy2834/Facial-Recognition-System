import cv2
import os
import csv
import numpy as np
from datetime import datetime
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Paths
dataset_folder = "dataset"
recognized_faces_folder = "recognized_faces"
attendance_file = "attendance.csv"
detection_log_file = "detection_log.txt"

# Ensure directories exist
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(recognized_faces_folder, exist_ok=True)

# Load dataset embeddings
precomputed_embeddings = {}

# Face recognition threshold (higher value = more strict match)
THRESHOLD = 0.7  

def compute_embedding(image_path):
    """Compute embedding of a given image."""
    try:
        img_representation = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
        return np.array(img_representation)
    except Exception as e:
        print(f"Error computing embedding for {image_path}: {e}")
        return None

def load_dataset_embeddings():
    """Load all embeddings from dataset folder."""
    precomputed_embeddings.clear()
    for person_name in os.listdir(dataset_folder):
        person_path = os.path.join(dataset_folder, person_name)
        if os.path.isdir(person_path):
            embeddings = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                embedding = compute_embedding(img_path)
                if embedding is not None:
                    embeddings.append(embedding)
            if embeddings:
                precomputed_embeddings[person_name] = embeddings  # Store multiple embeddings per person

def recognize_face(face_image):
    """Recognize face by comparing embeddings."""
    try:
        face_embedding = DeepFace.represent(face_image, model_name="Facenet", enforce_detection=True)[0]["embedding"]
    except:
        print("❌ Face detection failed. Try adjusting camera angle or lighting.")
        return None
    
    best_match = None
    best_similarity = 0
    
    for person_name, stored_embeddings in precomputed_embeddings.items():
        for stored_embedding in stored_embeddings:  # Compare against all stored embeddings
            similarity = cosine_similarity([face_embedding], [stored_embedding])[0][0]
            print(f"Checking {person_name} - Similarity: {similarity}")
            if similarity > THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name
    
    return best_match

def save_face(image, name, folder):
    """Save captured face image in a folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_folder = os.path.join(folder, name)
    os.makedirs(person_folder, exist_ok=True)
    img_count = len(os.listdir(person_folder)) + 1
    save_path = os.path.join(person_folder, f"{name}_{img_count}.jpg")
    cv2.imwrite(save_path, image)
    return save_path

def log_detection(name):
    """Log the recognized face in a text file."""
    with open(detection_log_file, "a") as f:
        f.write(f"{datetime.now()} - Recognized: {name}\n")
    print(f"✅ Log updated for {name}")

def update_attendance(name):
    """Update the attendance log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name])
    print(f"✅ Attendance updated for {name}")

def capture_and_recognize():
    """Capture live face and recognize."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return
    
    print("📸 Press 's' to capture an image or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture an image.")
            break
        
        cv2.imshow("Live Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            
            recognized_name = recognize_face(frame)
            
            if recognized_name:
                print(f"✅ Match Found! The person is {recognized_name}")
                save_face(frame, recognized_name, recognized_faces_folder)
                log_detection(recognized_name)
                update_attendance(recognized_name)
            else:
                new_name = input("❓ Unknown face detected. Enter name: ")
                save_path = save_face(frame, new_name, dataset_folder)
                save_face(frame, new_name, recognized_faces_folder)
                print(f"✅ New face saved as {new_name}")
                log_detection(new_name)
                update_attendance(new_name)
                precomputed_embeddings.setdefault(new_name, []).append(compute_embedding(save_path))
                load_dataset_embeddings()
            break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_dataset_embeddings()
    capture_and_recognize()


