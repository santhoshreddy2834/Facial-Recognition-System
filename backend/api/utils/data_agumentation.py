import numpy as np
import cv2
import random

def augment_face(image):
    augmented = image.copy()

    # Random horizontal flip
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    # Random brightness
    if random.random() > 0.5:
        factor = 1.0 + (random.random() - 0.5) * 0.4  # range: 0.8 to 1.2
        augmented = np.clip(augmented * factor, 0, 1)

    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        h, w = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (w, h))

    return augmented
