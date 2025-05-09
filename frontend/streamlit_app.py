import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Facial Recognition System", layout="wide")

# Navigation bar
st.sidebar.title("🔐 Facial Recognition Menu")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧑‍💼 Register Face", "🚨 Criminal Gallery", "📷 Live Face Recognition", "📊 Attendance"])

API_BASE = "http://localhost:5000"

def home_page():
    st.title("😎 Facial Recognition System")
    st.markdown("### 🔍 Real-Time Face Detection & Recognition")

    try:
        st.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/facial_recog.png", use_container_width=True)
    except Exception as e:
        st.warning("Image could not be loaded. Please check your internet connection or try again later.")

    st.markdown("##### Navigate through the sidebar to:")
    st.markdown("- 📷 Detect faces live from your webcam")
    st.markdown("- 🧑‍💼 Register new faces")
    st.markdown("- 🚨 Detect criminals in real-time")
    st.markdown("- 📊 View attendance records")

def register_face():
    st.title("🧑‍💼 Register New Face")
    name = st.text_input("Enter Name")
    image_file = st.file_uploader("Upload Face Image", type=['jpg', 'png'])

    if st.button("Register") and name and image_file:
        img = Image.open(image_file).convert("RGB")
        img_np = np.array(img)
        _, img_encoded = cv2.imencode('.jpg', img_np)
        response = requests.post(f"{API_BASE}/register", files={"file": img_encoded.tobytes()}, data={"name": name})

        if response.status_code == 200:
            st.success(f"Face registered successfully for {name}")
        else:
            st.error("Registration failed")

def view_criminal_gallery():
    st.title("🚨 Criminal Gallery")
    res = requests.get(f"{API_BASE}/criminals")
    if res.status_code == 200:
        criminals = res.json()["criminals"]
        for crim in criminals:
            st.image(crim["image_path"], caption=crim["name"], width=150)
    else:
        st.error("Unable to fetch criminal data")

def live_face_recognition():
    st.title("📷 Live Face Recognition")
    st.markdown("Press 's' to capture photo for recognition")

    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    run = st.checkbox('Start Webcam')

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        if st.button("📸 Scan Face"):
            _, img_encoded = cv2.imencode('.jpg', frame)
            response = requests.post(f"{API_BASE}/recognize", files={"file": img_encoded.tobytes()})
            if response.status_code == 200:
                result = response.json()
                if result['recognized']:
                    st.success(f"Detected: {result['name']}")
                    if result['is_criminal']:
                        st.error("🚨 This person is a CRIMINAL")
                else:
                    st.warning("No match found.")
            else:
                st.error("Recognition failed.")

    cap.release()

def attendance_view():
    st.title("📊 Attendance Records")
    res = requests.get(f"{API_BASE}/attendance")
    if res.status_code == 200:
        records = res.json().get("attendance", [])
        for rec in records:
            st.write(f"{rec['name']} - {rec['timestamp']}")
    else:
        st.error("Could not load attendance data.")

# Routing
if page == "🏠 Home":
    home_page()
elif page == "🧑‍💼 Register Face":
    register_face()
elif page == "🚨 Criminal Gallery":
    view_criminal_gallery()
elif page == "📷 Live Face Recognition":
    live_face_recognition()
elif page == "📊 Attendance":
    attendance_view()
