import os
import streamlit as st
from PIL import Image
import pandas as pd

def show_registered_faces(registered_dir):
    st.subheader("Registered Face Gallery")

    if not os.path.exists(registered_dir):
        st.warning("No registered faces found.")
        return

    for person in os.listdir(registered_dir):
        person_dir = os.path.join(registered_dir, person)
        if os.path.isdir(person_dir):
            st.markdown(f"**{person}**")
            images = os.listdir(person_dir)
            cols = st.columns(5)
            for i, img in enumerate(images):
                img_path = os.path.join(person_dir, img)
                try:
                    with Image.open(img_path) as image:
                        cols[i % 5].image(image, width=100)
                except Exception as e:
                    st.error(f"Could not load image: {img_path}")

def show_attendance_log(log_file="utils/detection_log/attendance_log.csv"):
    st.subheader("Attendance Records")

    if not os.path.exists(log_file):
        st.warning("No attendance log found.")
        return

    try:
        df = pd.read_csv(log_file)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading attendance log: {e}")
