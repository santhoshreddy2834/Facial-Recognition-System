import cv2
from deepface import DeepFace

# Load reference image
reference_image = "dataset/sample1.jpg"

# Initialize webcam
cap = cv2.VideoCapture(0)

print("🎥 Starting webcam... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't capture video frame!")
        break

    # Display the video frame
    cv2.imshow("Webcam Feed", frame)

    # Press 'r' to recognize the face
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        try:
            print("🔍 Recognizing face...")
            result = DeepFace.verify(img1_path=reference_image, img2_path=frame, model_name="Facenet")
            
            if result["verified"]:
                print("✅ Match Found!")
            else:
                print("❌ No Match!")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
