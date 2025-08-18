import os
import cv2 
import numpy as np 
import streamlit as st 
from insightface.app import FaceAnalysis 
from sklearn.metrics.pairwise import cosine_similarity
import logging


# ----------------- Face Analysis Setup -----------------
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

EMPLOYEE_DIR = "employees"

def load_employee_image(empid):
    """Load the stored image for a given employee ID."""
    for ext in ["jpg", "jpeg", "png"]:
        path = os.path.join(EMPLOYEE_DIR, f"{empid}.{ext}")
        if os.path.exists(path):
            return cv2.imread(path)
    return None

def get_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None, None
    return faces[0].normed_embedding, faces[0]

def verify_employee(empid, input_image, threshold=0.65):
    """Verify input image against the stored image of empid."""
    stored_img = load_employee_image(empid)
    if stored_img is None:
        return None, "Employee ID not found in database"

    stored_emb, _ = get_embedding(stored_img)
    input_emb, _ = get_embedding(input_image)

    if stored_emb is None or input_emb is None:
        return None, "Face not detected."

    # Cosine similarity
    sim = cosine_similarity([stored_emb], [input_emb])[0][0]
    verified = sim > threshold



    return verified, sim

# ----------------- Streamlit UI -----------------
st.title("Employee Face Verification System")
st.write("Verify uploaded image OR live webcam feed against employee database")

empid = st.text_input("Enter Employee ID:")

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Webcam"])

# ---------- Tab 1: Upload ----------
with tab1:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and empid:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result, message = verify_employee(empid, img)

        if result is None:
            st.error(message)
            
        else:
            if result:
                st.success("Verification Passed.")
            else:
                st.error("Verification Failed.")

# ---------- Tab 2: Live Webcam ----------
with tab2:
    st.write("Click below to capture from webcam and verify:")

    # Live webcam input
    live_frame = st.camera_input("Capture Face")

    if live_frame and empid:
        file_bytes = np.asarray(bytearray(live_frame.read()), dtype=np.uint8)
        live_img = cv2.imdecode(file_bytes, 1)

        result, message = verify_employee(empid, live_img)

        if result is None:
            st.error(message)
            
        else:
            if result:
                st.success("Verification Passed.")
            else:
                st.error("Verification Failed.")
