from deepface import DeepFace
import cv2
import os

# Paths
reference_image = "dataset/reference.jpg"
test_image = "dataset/sample1.jpg"

try:
    # Check if images exist
    if not os.path.exists(reference_image) or not os.path.exists(test_image):
        raise FileNotFoundError("❌ Error: Reference or test image not found!")

    # Load images
    ref_img = cv2.imread(reference_image)
    test_img = cv2.imread(test_image)

    if ref_img is None or test_img is None:
        raise ValueError("❌ Error: One or both images failed to load!")

    # Perform face verification (compare both images)
    result = DeepFace.verify(img1_path=reference_image, img2_path=test_image, model_name="Facenet")

    # Print results
    if result["verified"]:
        print("✅ Match Found! The person in the test image is the same as the reference image.")
    else:
        print("❌ No Match! The person in the test image is different.")

except Exception as e:
    print(f"❌ Error: {e}")

