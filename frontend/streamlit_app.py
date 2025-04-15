import streamlit as st
import requests
import cv2
import numpy as np
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Facial Recognition System", layout="wide")

st.title("🎥 Facial Recognition System Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("Select Option", ["🔍 Live Face Recognition", "📥 Register New Face", "📂 Registered Faces", "🧾 Attendance Logs", "🔁 Update Embeddings"])

# Capture Image Helper
def capture_image():
    cap = cv2.VideoCapture(0)
    st.info("📸 Press 's' to capture, 'q' to quit.")
    captured_image = None

    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera - Press 's' to Capture", frame)

        key = cv2.waitKey(1)
        if key == ord("s"):
            captured_image = frame
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

# Live Face Recognition
if page == "🔍 Live Face Recognition":
    st.subheader("🔍 Real-Time Face Recognition")
    if st.button("📷 Capture & Recognize"):
        frame = capture_image()
        if frame is not None:
            _, img_encoded = cv2.imencode('.jpg', frame)
            try:
                response = requests.post(f"{API_URL}/recognize", files={"image": img_encoded.tobytes()})
                result = response.json()
                st.success(f"👤 Recognized: {result['name']}")
                st.info(f"🎯 Confidence: {result['confidence']}%")
                st.warning(f"🚨 Criminal: {result['is_criminal']}")
                st.image(frame, channels="BGR", caption="Captured Face")
            except Exception as e:
                st.error(f"❌ Error contacting backend: {e}")

# Register New Face
elif page == "📥 Register New Face":
    st.subheader("📥 Register a New Face")
    name = st.text_input("Full Name")
    is_criminal = st.selectbox("Criminal Status", ["no", "yes"])

    if st.button("📷 Capture & Register"):
        frame = capture_image()
        if frame is not None:
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {"image": img_encoded.tobytes()}
            data = {"name": name, "is_criminal": is_criminal}
            try:
                response = requests.post(f"{API_URL}/register", data=data, files=files)
                st.success(response.json()["message"])
                st.image(frame, channels="BGR", caption="Registered Face")
            except Exception as e:
                st.error(f"❌ Registration failed: {e}")

# Registered Faces
elif page == "📂 Registered Faces":
    st.subheader("📂 Registered Faces")
    try:
        response = requests.get(f"{API_URL}/registered_faces")
        face_data = response.json()
        cols = st.columns(4)
        for i, person in enumerate(face_data):
            with cols[i % 4]:
                img_path = f"../data/faces/{person['image']}"
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=person["name"], use_column_width=True)
    except Exception as e:
        st.error(f"❌ Failed to load faces: {e}")

# Attendance Logs
elif page == "🧾 Attendance Logs":
    st.subheader("🧾 Attendance Logs")
    try:
        response = requests.get(f"{API_URL}/attendance_logs")
        logs = response.json()
        for log in logs[::-1]:
            st.markdown(f"👤 **{log['name']}** | 🕒 {log['time']} | 🚨 Criminal: {log['is_criminal']}")
    except Exception as e:
        st.error(f"❌ Failed to fetch logs: {e}")

# Update Embeddings
elif page == "🔁 Update Embeddings":
    st.subheader("🔁 Refresh Face Embeddings")
    if st.button("🔄 Update Now"):
        try:
            response = requests.post(f"{API_URL}/update_embeddings")
            st.success(response.json()["message"])
        except Exception as e:
            st.error(f"❌ Failed to update: {e}")
