import numpy as np

def find_matching_face(embedding, face_db, threshold=0.7):
    min_dist = float("inf")
    identity = "Unknown"
    criminal_status = "Unknown"

    for name, data in face_db.items():
        db_embedding = data["embedding"]
        dist = np.linalg.norm(embedding - db_embedding)

        if dist < min_dist:
            min_dist = dist
            if dist < threshold:
                identity = name
                criminal_status = data.get("criminal", "Unknown")

    return identity, min_dist, criminal_status
