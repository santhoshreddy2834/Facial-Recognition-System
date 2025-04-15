# utils/api_handler.py

import requests

BASE_URL = "http://127.0.0.1:5000"

def register_face_api(name, criminal):
    try:
        res = requests.post(f"{BASE_URL}/register", json={
            "name": name,
            "criminal": criminal
        })
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def recognize_face_api():
    try:
        res = requests.post(f"{BASE_URL}/recognize")
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def get_registered_faces_api():
    try:
        res = requests.get(f"{BASE_URL}/registered_faces")
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def get_attendance_logs_api():
    try:
        res = requests.get(f"{BASE_URL}/attendance_logs")
        return res.json()
    except Exception as e:
        return {"error": str(e)}
