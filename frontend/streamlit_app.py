import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# Flask API URL
API_URL = "http://127.0.0.1:5000"

st.title("🔍 Facial Recognition System")
st.sidebar.header("⚙️ Options")

# Upload an image for recognition
uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send to Flask API for recognition
    st.write("⏳ Processing...")
    response = requests.post(f"{API_URL}/recognize", files={"file": uploaded_file})

    if response.status_code == 200:
        result = response.json()
        if result["recognized"]:
            st.success(f"✅ Recognized: {result['name']}")
            st.image(result["image_url"], caption="Matched Face", use_column_width=True)
        else:
            st.warning("❌ No match found. Face not recognized!")
    else:
        st.error("⚠️ Error in face recognition!")

# Show latest logs
if st.sidebar.button("📜 View Detection Log"):
    logs_response = requests.get(f"{API_URL}/logs")
    if logs_response.status_code == 200:
        logs = logs_response.json().get("logs", [])
        st.sidebar.text_area("📜 Detection Log", "\n".join(logs), height=200)
    else:
        st.sidebar.error("⚠️ Unable to fetch logs!")

# Run live recognition (future feature)
if st.sidebar.button("📷 Start Live Recognition"):
    st.sidebar.warning("🚀 Live camera integration coming soon!")

st.sidebar.text("🔒 Secure AI-Powered System")
