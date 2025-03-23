import cv2
import os

# Folder where dataset images will be stored
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Capture live image
cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)
ret, frame = cap.read()

if ret:
    name = input("Enter person's name: ")  # Get name input
    filename = os.path.join(dataset_path, f"{name}.jpg")
    
    cv2.imwrite(filename, frame)  # Save image
    print(f"✅ Image captured and saved as {filename}")
else:
    print("❌ Failed to capture image!")

cap.release()
cv2.destroyAllWindows()

