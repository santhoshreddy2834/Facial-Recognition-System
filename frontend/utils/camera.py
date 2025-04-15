# utils/camera.py

import cv2
import os
import time

def capture_image(save_path, name, num_images=10):
    cap = cv2.VideoCapture(0)
    count = 0
    os.makedirs(save_path, exist_ok=True)
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_name = os.path.join(save_path, f"{name}_{count}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

        cv2.imshow("Capturing Faces - Press Q to Exit", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
